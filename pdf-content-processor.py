#!/usr/bin/env python

import argparse
import json
import logging
import os
import re
import sys

import astrapy as ap
import llama_parse as lp
import PyPDF2 as pp2
import yaml


class PdfContentProcessor:
    def __init__(self, logger=None, **parser_config_kwargs: dict):
        self._logger = logger or logging.getLogger(self.__class__.__name__)

        self.parsing_token_key_path = ''
        self.parsing_instructions_path = ''
        self.parsing_retries = 3

        self.astra_token_json_path = ''
        self.astra_db_id = ''
        self.astra_db_region = ''
        self.astra_db_keyspace = ''
        self.astra_db_table = ''

        self.__dict__.update(parser_config_kwargs)


    def _initialise_document_parser(self, source_index_path) -> 'DocumentParser' or None:
        if os.path.exists(source_index_path):
            return None
        else:
            return DocumentParser(self.parsing_token_key_path, self.parsing_instructions_path, self.parsing_retries)

    def _initialise_document_loader(self) -> 'DocumentLoader':
        return DocumentLoader(
            self.astra_token_json_path,
            self.astra_db_id,
            self.astra_db_region,
            self.astra_db_keyspace,
            self.astra_db_table
        )

    def _delete_arg_to_bool(self, delete_output: str) -> (bool, bool):
        if delete_output:
            if delete_output == 'source':
                return True, False
            elif delete_output == 'markdown':
                return False, True
            elif delete_output == 'all':
                return True, True
            else:
                self._logger.error(f'Invalid value for delete_output: {delete_output}')
                exit(1)

        return False, False

    def _prepare_dir(self, dir_path: str, index_file: str, delete_output: bool, delete_extn=None) -> None:
        if os.path.exists(dir_path):
            if delete_output or not os.path.exists(index_file):
                self._logger.info(f'Deleting contents of {dir_path}')
                self._delete_dir_contents(dir_path, delete_extn=delete_extn)
        else:
            os.makedirs(dir_path, exist_ok=True)

    @staticmethod
    def _delete_dir_contents(dir_path: str, delete_extn=None) -> None:
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                if delete_extn:
                    if os.path.splitext(item_path)[1] in delete_extn:
                        os.unlink(item_path)
                else:
                    os.unlink(item_path)
            elif os.path.isdir(item_path):
                PdfContentProcessor._delete_dir_contents(item_path)
                os.rmdir(item_path)

    @staticmethod
    def main_cli() -> None:
        arg_parser = argparse.ArgumentParser(
            description='Processes a PDF file to extract its information using LlamaParse into markdown, and'
                        ' optionally load into an Astra Vector Database.'
        )
        arg_parser.add_argument('document_path', help='Path to the PDF file to parse.')
        arg_parser.add_argument(
            '--parsing_token_key_path',
            help='Path to the file containing the API token.'
        )
        arg_parser.add_argument(
            '--parsing_instructions_path',
            help='Path to text file containing the parsing instructions.'
        )
        arg_parser.add_argument(
            '--parsing_retries',
            help='Number of times to attempt to parse a document before giving up. Default is 3.',
            type=int,
            default=3
        )
        arg_parser.add_argument(
            '--astra_token_json_path',
            help='Path to the file containing the Astra API JSON token.'
        )
        arg_parser.add_argument('--astra_db_id', help='Astra database ID.')
        arg_parser.add_argument('--astra_db_region', help='Astra region.')
        arg_parser.add_argument('--astra_db_keyspace', help='Astra keyspace.')
        arg_parser.add_argument('--astra_db_table', help='Astra table.')
        arg_parser.add_argument(
            '-a',
            '--abort_on_failure',
            help='Flag to abort processing of document if parsing fails after retrying.',
            action='store_true'
        )
        arg_parser.add_argument(
            '-d',
            '--delete',
            help='Flag to delete content and information output generate. Output is generated in the \'source\' and '
                 ' \'markdown\' directories. A single directory or both directories can be deleted by specifying a' 
                 ' value with this option. The value can be either; \'source\', \'markdown\', or \'all\'. '
                 ' If this flag is unspecified, and the source and markdown indexes are present in their respective '
                 ' directories, the contents in the directories are preserved.',
            choices=['source', 'markdown', 'all']
        )
        arg_parser.add_argument(
            '-i',
            '--extract_images',
            help='Flag to extract images from the document.',
            action='store_true'
        )
        arg_parser.add_argument(
            '-l',
            '--load_database',
            help='Flag to load the markdown generated into Astra.',
            action='store_true'
        )
        arg_parser.add_argument(
            '-p',
            '--pages_per_split',
            help='When splitting the document before LLM parsing, this is the maximum number of pages within each '
                 ' split document. Default is 50.',
            type=int,
            default=50
        )
        arg_parser.add_argument(
            '-r',
            '--row_size_bytes',
            help='Maximum row size in bytes used to generate embeddings for a page. If a document page is bigger than'
                 ' this size, it will be split before it is loaded into Astra. Default is 8000.',
            type=int,
            default=8000
        )

        parsed_args = arg_parser.parse_args()

        processor_configs = {
            'parsing_token_key_path': parsed_args.parsing_token_key_path,
            'parsing_instructions_path': parsed_args.parsing_instructions_path,
            'parsing_retries': parsed_args.parsing_retries,
            'astra_token_json_path': parsed_args.astra_token_json_path,
            'astra_db_id': parsed_args.astra_db_id,
            'astra_db_region': parsed_args.astra_db_region,
            'astra_db_keyspace': parsed_args.astra_db_keyspace,
            'astra_db_table': parsed_args.astra_db_table,
        }

        pdf_content_processor = PdfContentProcessor(**processor_configs)
        pdf_content_processor.process_document(
            parsed_args.document_path,
            delete_output=parsed_args.delete,
            abort_on_failure=parsed_args.abort_on_failure,
            extract_images=parsed_args.extract_images,
            pages_per_split=parsed_args.pages_per_split,
            load_database=parsed_args.load_database,
            row_size_bytes=parsed_args.row_size_bytes
        )

    def process_document(
            self,
            document_path: str,
            delete_output=None,
            abort_on_failure=False,
            extract_images=False,
            pages_per_split=50,
            load_database=False,
            row_size_bytes=8000
    ) -> None:
        self._logger.info(f"Using Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

        self._logger.info(f'Processing document {document_path}')
        output_dir = '.'.join(document_path.split('.')[:-1])

        self._logger.info(f'Generated artifacts will be stored in {output_dir}')
        base_name = output_dir.split('/')[-1]
        source_dir = os.path.join(output_dir, 'source')
        markdown_dir = os.path.join(output_dir, 'markdown')

        source_index_path = os.path.join(source_dir, SourceIndex.file_name)
        markdown_index_path = os.path.join(markdown_dir, MarkdownIndex.file_name)

        # Initialise the document parser and if required the document loader now. This is so we can validate the
        # parser arguments before start any directory setup or document processing.
        document_parser = self._initialise_document_parser(source_index_path)
        if load_database:
            document_loader = self._initialise_document_loader()
        else:
            document_loader = None

        delete_source, delete_markdown = self._delete_arg_to_bool(delete_output)

        self._prepare_dir(source_dir, source_index_path, delete_source)
        self._prepare_dir(markdown_dir, markdown_index_path, delete_markdown, delete_extn=['.md', '.yaml'])

        if os.path.exists(source_index_path):
            self._logger.info(f'Loading source index from file {source_index_path}')
            source_index = SourceIndex.load(source_index_path)
        else:
            source_index = SourceIndex(base_name=base_name, source_dir=source_dir)
            self._logger.info('Splitting document into smaller documents')
            document_splitter = DocumentSplitter(source_index)
            document_splitter.split_document(document_path, source_dir, pages_per_split)

            document_parser.parse(source_index, extract_images, abort_on_failure)
            source_index.save()

        if os.path.exists(markdown_index_path):
            self._logger.info(f'Loading markdown index from file {markdown_index_path}')
            markdown_index = MarkdownIndex.load(markdown_index_path)
        else:
            markdown_index = MarkdownIndex(base_name=base_name, markdown_dir=markdown_dir)
            if not source_index:
                self._logger.info(f'Loading source index from file {source_index_path}')
                source_index = SourceIndex.load(source_index_path)

            markdown_parser = RawContentParser()
            markdown_parser.parse(source_index, markdown_index)
            markdown_index.save()

        if document_loader:
            if not markdown_index:
                self._logger.info(f'Loading markdown index from file {markdown_index_path}')
                markdown_index = MarkdownIndex.load(markdown_index_path)
            document_name = os.path.basename(document_path)
            document_loader.load_markdown(markdown_index, document_name, row_size_bytes)

        self._logger.info('Processing complete!')

    @staticmethod
    def write_markdown_page(file_path: str, markdown_output: str) -> None:
        with open(file_path, 'w', encoding='utf-8') as file_out_h:
            file_out_h.write(markdown_output)
            file_out_h.write('\n\n')

    @staticmethod
    def process_images(images_dir_path: str) -> list:
        image_files = []
        if images_dir_path:
            image_files = [
                f for f in os.listdir(images_dir_path) if os.path.isfile(os.path.join(images_dir_path, f))
            ]
            image_files.sort()

        return image_files


class OutputIndex:
    def __init__(self,  base_name, output_dir):
        self._base_name = base_name
        self._output_dir = output_dir

    def base_name(self) -> str:
        return self._base_name

    def dir(self) -> str:
        return self._output_dir

    @staticmethod
    def load(index_file: str) -> 'OutputIndex':
        raise NotImplemented

    def save(self) -> None:
        raise NotImplemented

class SourceIndex(OutputIndex):
    file_name = 'source_index.yaml'

    def __init__(self, logger=None, base_name=None, source_dir=None, document_extn='.pdf'):
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        super().__init__(base_name, source_dir)
        self.document_extn = document_extn

        self._source_ranges = []
        self._source_range_info = {}

    def __iter__(self):
        for range_name in self._source_ranges:
            yield range_name, self._source_range_info[range_name]

    def append_page_range(self, start_page: int, end_page: int) -> (str, str):
        range_name = f'pages_{start_page}-{end_page}'
        raw_file = f'{self._base_name}_{range_name}{self.document_extn}'
        self._source_ranges.append(range_name)
        self._source_range_info[range_name] = {
            'file': raw_file,
            'range': (start_page, end_page)
        }

        return range_name, raw_file

    # YAML file is in the format
    # ---
    # source:
    #   base_name: <basename>
    #   output_dir: <output_directory>
    #   raw:
    #     - pages_<start_page>-<end_page>:
    #       file: <split_filename>
    #       range:
    #         start: <start_page>
    #         end: <end_page>
    @staticmethod
    def load(index_file: str) -> OutputIndex:
        with open(index_file, 'r', encoding='utf-8') as index_file_h:
            index_data = yaml.safe_load(index_file_h)

            source_info = index_data['source']
            raw_info = source_info['raw']
            source_index = SourceIndex(base_name=source_info['base_name'], source_dir=source_info['output_dir'])

            for page_range in raw_info:
                range_name = list(page_range.keys())[0]
                range_info = page_range[range_name]

                source_index.append_page_range(range_info['range']['start'], range_info['range']['end'])

            return source_index

    def save(self) -> None:
        index_data = {
            'source': {
                'base_name': self._base_name,
                'output_dir': self._output_dir,
                'raw': [],
            }
        }

        for range_name in self._source_ranges:
            range_info = self._source_range_info[range_name]
            index_data['source']['raw'].append({
                range_name: {
                    'file': range_info['file'],
                    'range': {
                        'start': range_info['range'][0],
                        'end': range_info['range'][1]
                    }
                }
            })

        index_path = os.path.join(self._output_dir, SourceIndex.file_name)
        self._logger.info(f'Saving source index to {index_path}')
        with open(index_path, 'w', encoding='utf-8') as index_file_h:
            yaml.dump(index_data, index_file_h)


class MarkdownIndex(OutputIndex):
    file_name = 'markdown_index.yaml'

    def __init__(self, logger=None, title='', base_name=None, markdown_dir=None):
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        super().__init__(base_name, markdown_dir)

        self._document_title = title
        self._markdown_pages = []

    def __iter__(self):
        for md_page in self._markdown_pages:
            yield md_page['page'], md_page['file']

    def set_title(self, doc_title) -> None:
        if self._document_title:
            self._logger.warning(f'Unable to set title to {doc_title}. Title already set to {self._document_title}')
            return
        self._document_title = doc_title

    def get_title(self) -> str:
        return self._document_title

    def append_page(self, page_num: int) -> str:
        markdown_file = f'{self._base_name}_page_{str(page_num).zfill(3)}.md'
        self._markdown_pages.append({
            'page': page_num,
            'file': markdown_file
        })

        return markdown_file

    # YAML file is in the format
    # ---
    # markdown:
    #   base_name: <basename>
    #   output_dir: <output_directory>
    #   title: <document_title>
    #   pages:
    #     - page: <page_num>
    #       file: <split_filename>
    @staticmethod
    def load(index_file: str) -> OutputIndex:
        with open(index_file, 'r', encoding='utf-8') as index_file_h:
            index_data = yaml.safe_load(index_file_h)

            markdown_info = index_data['markdown']
            markdown_index = MarkdownIndex(
                title=markdown_info['title'],
                base_name=markdown_info['base_name'],
                markdown_dir=markdown_info['output_dir']
            )

            for page_info in markdown_info['pages']:
                markdown_index.append_page(page_info['page'])

            return markdown_index

    def save(self) -> None:
        index_data = {
            'markdown': {
                'title': self._document_title,
                'base_name': self._base_name,
                'output_dir': self._output_dir,
                'pages': [] + self._markdown_pages
            }
        }

        index_path = os.path.join(self._output_dir, MarkdownIndex.file_name)
        self._logger.info(f'Saving markdown index to {index_path}')
        with open(index_path, 'w', encoding='utf-8') as index_file_h:
            yaml.dump(index_data, index_file_h)

class DocumentSplitter:
    def __init__(self, source_index: SourceIndex, logger=None):
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self.source_index = source_index

    def split_document(self, document_path: str, output_parent_dir: str, pages_per_split=50) -> None:
        pdf_reader = pp2.PdfReader(document_path)

        num_pages_total = len(pdf_reader.pages)
        num_documents = (num_pages_total + pages_per_split - 1) // pages_per_split

        for doc_index in range(num_documents):
            start_page = doc_index * pages_per_split
            end_page = min(start_page + pages_per_split, num_pages_total)

            range_name, output_doc_name = self.source_index.append_page_range(start_page + 1, end_page)

            range_dir = os.path.join(output_parent_dir, range_name)
            output_doc_path = os.path.join(range_dir, output_doc_name)

            os.makedirs(range_dir, exist_ok=True)

            if os.path.exists(output_doc_path):
                self._logger.warning(f'Skipping {output_doc_path} as it already exists.')
                continue

            pdf_writer = pp2.PdfWriter()
            for page_num in range(start_page, end_page):
                pdf_writer.add_page(pdf_reader.pages[page_num])

            with open(output_doc_path, "wb") as output_pdf:
                pdf_writer.write(output_pdf)

            self._logger.info(f'Created {output_doc_path} containing pages {start_page + 1} to {end_page}.')


class DocumentParser:
    def __init__(self, token_file: str, instructions_file: str, retry_count=3, logger=None):
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._llama_parse_parser = None
        self._parsing_retries = retry_count

        self._initialise(token_file, instructions_file)

    def _initialise(self, token_file: str, instructions_file: str) -> None:
        llama_parse_api_key = ''
        llama_parse_instructions = ''

        if token_file:
            with open(token_file, 'r') as file_h:
                llama_parse_api_key = file_h.read().strip()
        else:
            self._logger.error('No LlamaParse token provided. Token required to parse data.')
            exit(1)

        if instructions_file:
            with open(instructions_file, 'r') as file_h:
                llama_parse_instructions = ''.join(line for line in file_h if not line.startswith('#'))
        else:
            self._logger.warning(
                'No parsing instructions provided. This will likely result in the document being parsed incorrectly.')
            while True:
                response = input('Do you want to continue? (Y/N): ').strip().lower()
                if response:
                    if response[0] == 'y':
                        break
                    elif response[0] == 'n':
                        exit(0)

                print("Invalid input. Please enter 'Y' to continue, or 'N' to abort.")

        self._llama_parse_parser = lp.LlamaParse(
            api_key=llama_parse_api_key,
            result_type=lp.ResultType.MD,
            parsing_instruction=llama_parse_instructions,
            verbose=True,
            invalidate_cache=True,
            do_not_cache=True,
        )

    def _generate_raw_output(
            self,
            document_path: str,
            abort_on_failure: bool,
            markdown_raw_dir: str,
            base_name: str,
            images_dir: str
    ) -> None:
        document_content = self._extract_document_content_with_llm(document_path, abort_on_failure)

        if not document_content:
            return

        self._logger.info(document_content[0]['job_metadata'])

        for extracted_content in document_content[0]['pages']:
            extracted_markdown = self._remove_markdown_markers_and_whitespace(extracted_content['md'])
            markdown_raw_file_name = '{}_page_{}.md'.format(base_name, str(extracted_content['page']).zfill(3))
            markdown_raw_file_path = os.path.join(markdown_raw_dir, markdown_raw_file_name)

            PdfContentProcessor.write_markdown_page(markdown_raw_file_path, extracted_markdown)

            if images_dir:
                self._llama_parse_parser.get_images(
                    document_content,
                    download_path=images_dir
                )

    def _extract_document_content_with_llm(self, document_path: str, abort_on_failure: bool) -> list or None:
        self._logger.info(f'Parsing document {document_path}')
        retry_count = 0
        job_response = []
        while retry_count < self._parsing_retries:
            # Attempt to parse the document using the LlamaParse API
            job_response = self._llama_parse_parser.get_json_result(document_path)

            if len(job_response) > 0:
                break

            self._logger.info('Failed to parse document. Retrying...')
            retry_count += 1

        if len(job_response) < 1:
            if abort_on_failure:
                self._logger.error('Failed to parse document after {} retries. Aborting.'.format(self._parsing_retries))
                exit(1)
            else:
                self._logger.warning('Failed to parse document after {} retries. Skipping.'.format(self._parsing_retries))
                return None

        return job_response

    @staticmethod
    def _remove_markdown_markers_and_whitespace(page_markdown: str) -> str:
        processed_page = page_markdown

        if processed_page.startswith('```markdown'):
            processed_page = processed_page[len('```markdown'):]

        if processed_page.endswith('```'):
            processed_page = processed_page[:-len('```')]

        return processed_page.strip()

    # Extracts content in document using LlamaParse, into raw markdown pages which are then written to disk
    def parse(self, source_index: SourceIndex, extract_images: bool, abort_on_failure: bool) -> None:
        source_dir = source_index.dir()

        for range_name, range_info in source_index:
            source_part_base_dir = os.path.join(source_dir, range_name)
            source_part_raw_dir = os.path.join(source_part_base_dir, 'raw')
            source_part_doc = os.path.join(source_part_base_dir, range_info['file'])

            os.makedirs(source_part_raw_dir, exist_ok=True)
            if extract_images:
                source_part_image_dir = os.path.join(source_part_base_dir, 'images')
                os.makedirs(source_part_image_dir, exist_ok=True)
            else:
                source_part_image_dir = None

            self._generate_raw_output(
                source_part_doc,
                abort_on_failure,
                source_part_raw_dir,
                source_index.base_name(),
                source_part_image_dir
            )


class RawContentParserContext:
    def __init__(self, logger, markdown_index: MarkdownIndex, start_page: int, end_page: int):
        self._logger = logger
        self._markdown_index = markdown_index
        self._page_num = start_page
        self._pages_remaining = end_page - start_page
        self._page_type = None
        self._line_buffer = []

    def __int__(self):
        return self._page_num

    def new_page(self, page_type: str) -> None:
        if not self._page_type is None:
            self._page_num += 1
            self._pages_remaining -= 1

        self._page_type = page_type
        self._line_buffer = []

    def append_line(self, line: str) -> None:
        if self._page_type == 'information':
            self._line_buffer.append(line)
        elif self._page_type == 'title':
            self._line_buffer.append(line.strip())

    def end_page(self):
        if self._page_type is None:
            return []

        if self._page_type == 'information':
            if self._line_buffer:
                return [MarkdownPage(self._page_num, '\n'.join(self._line_buffer))]
        elif self._page_type == 'title':
            doc_title = ' '.join(self._line_buffer).title().strip()
            self._logger.info(f'Found document title \'{doc_title}\'. Saving ...')
            self._markdown_index.set_title(doc_title)
        elif self._page_type in ['contents', 'blank']:
            self._logger.info(f'Skipping page {self._page_num} as it is a \'{self._page_type}\' page')
        else:
            self._logger.warning(f'Skipping page {self._page_num} as \'{self._page_type}\' is an unknown page type!')

        return []

    def pages_remaining(self):
        return self._pages_remaining


class MarkdownPage:
    def __init__(self, page_num: int, markdown: str):
        self._page_num = page_num
        self._markdown = markdown

    def __str__(self):
        return self._markdown

    def __int__(self):
        return self._page_num


class RawContentParser:
    page_pattern = r'^\[(title|contents|information|blank)\]: #$'

    def __init__(self, logger=None):
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._markdown_pages = []

    @staticmethod
    def _get_source_raw_pages(range_name: str, source_dir: str) -> list:
        range_dir = os.path.join(source_dir, range_name, 'raw')
        markdown_raw_pages = []

        for raw_page in os.listdir(range_dir):
            raw_page_full_path = os.path.join(range_dir, raw_page)
            if os.path.isfile(raw_page_full_path) and raw_page.endswith('.md'):
                markdown_raw_pages.append(raw_page_full_path)

        markdown_raw_pages.sort()

        return markdown_raw_pages

    @staticmethod
    def _get_plurality(count: int, singular: str, plural: str) -> str:
        return singular if count == 1 else plural

    @staticmethod
    def _parse_raw_page(parser_ctxt: RawContentParserContext, raw_markdown: str, last_page: bool) -> list:
        # The parser context keeps track of the current page number so we know where we are up to in the document
        # parsing. This allows it to handle the case where the content is split over separate raw markdown pages
        raw_markdown_lines = raw_markdown.split('\n')
        markdown_pages = []

        # The parsing of the markdown relies heavily on the instructions passed to LlamaParse. For the parsing to work
        # correctly, LlamaParse must be instructed to place the page type at the very start of the markdown for that
        # page.
        for line in raw_markdown_lines:
            page_type_match = re.match(RawContentParser.page_pattern, line)
            if page_type_match:
                # Depending on the content size, LlamaParse will sometime combine the markdown for multiple pages into
                # one raw markdown page. We want to split them back into individual pages so the source of the
                # information can be easily traced back. If we hit a new page match while we are still processing the
                # raw page, we need to save what we have parsed before we start processing the new page.
                markdown_pages += parser_ctxt.end_page()
                parser_ctxt.new_page(page_type_match.groups()[0])

                continue
            # End page_type_match handling

            parser_ctxt.append_line(line)
        # End raw_markdown_lines for loop

        # We may still have lines (raw markdown) in the parser context buffer. These lines may be a complete page, or
        # they may be part of a page that flows into the next raw markdown page. Where the current page ends will be
        # defined by the next raw markdown page that is parsed. If this is the last raw markdown page in the loop, we
        # will save the content in the context buffer as a page after parsing all the raw pages in the range.
        if last_page:
            markdown_pages += parser_ctxt.end_page()

        return markdown_pages

    def _parse_range(self, range_name: str, range_info: dict, source_dir: str, markdown_index: MarkdownIndex) -> None:
        raw_pages_list = self._get_source_raw_pages(range_name, source_dir)
        raw_pages_count = len(raw_pages_list)
        markdown_dir = markdown_index.dir()

        # range_info is a dict with the following format:
        # {
        #      'file': file_name,
        #      'range': (start_page, end_page)
        # }
        start_page = range_info['range'][0]
        end_page = range_info['range'][1]
        parser_ctxt = RawContentParserContext(self._logger, markdown_index, start_page, end_page)

        self._logger.info(f'Parsing raw markdown content for range {range_name}')

        for raw_page_num, raw_page_path in enumerate(raw_pages_list):
            with open(raw_page_path, 'r', encoding='utf-8') as raw_page_file_h:
                raw_markdown = raw_page_file_h.read()

                self._logger.info(f'Parsing raw markdown content in page {raw_page_path}')
                extracted_markdown_pages = self._parse_raw_page(
                    parser_ctxt,
                    raw_markdown,
                    raw_page_num == raw_pages_count - 1
                )

            # Write out the markdown as we go along as some of the document pages might be large
            for markdown_page in extracted_markdown_pages:
                markdown_page_name = markdown_index.append_page(int(markdown_page))
                markdown_page_path = os.path.join(markdown_dir, markdown_page_name)
                self._logger.info(f'Writing markdown page {markdown_page_path}')
                PdfContentProcessor.write_markdown_page(markdown_page_path, str(markdown_page))

        pages_remaining = parser_ctxt.pages_remaining()
        if pages_remaining:
            expected_page_count = end_page - start_page + 1
            actual_page_count = expected_page_count - pages_remaining

            remaining_plural = RawContentParser._get_plurality(pages_remaining, 'page', 'pages')
            actual_plural = RawContentParser._get_plurality(actual_page_count, 'page', 'pages')
            expected_plural = RawContentParser._get_plurality(expected_page_count, 'page', 'pages')
            self._logger.warning(
                f'Markdown for {pages_remaining} {remaining_plural} is missing. Completed parsing all raw files for '
                f' {range_name}, but only created {actual_page_count} {actual_plural}. Expected {expected_page_count}'
                f' {expected_plural} to be created.'
            )

    # Parsing of the raw markdown pages written to disk; this will piece together the title, split and group page
    # information according to how the original document was structured, and determine the document page number that
    # information was extracted from.
    def parse(self, source_index: SourceIndex, markdown_index: MarkdownIndex) -> None:
        source_dir = source_index.dir()
        for source_range in source_index:
            range_name, range_info = source_range

            self._parse_range(range_name, range_info, source_dir, markdown_index)


class DocumentLoader:
    astra_domain = 'apps.astra.datastax.com'

    def __init__(self, token_file: str, db_id: str, db_region: str, keyspace_name: str, table_name: str, logger=None):
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._astra_token = ''
        self._astra_db_id = ''
        self._astra_db_region = ''
        self._astra_db_keyspace = ''
        self._astra_db_table = ''
        self._astra_api_endpoint = ''

        self._initialise(token_file, db_id, db_region, keyspace_name, table_name)

    def _initialise(self, token_file: str, db_id: str, db_region: str, ks_name: str, tbl_name: str) -> None:
        if token_file:
            with open(token_file, 'r') as file_h:
                astra_token = json.loads(file_h.read())
                self._astra_token = astra_token['token']
        else:
            self._logger.error('No Astra token provided. Token required to load data.')
            exit(1)

        if db_id:
            self._astra_db_id = db_id
        else:
            self._logger.error('No Astra database ID provided. Database ID required to load data.')
            exit(1)

        if db_region:
            self._astra_db_region = db_region
        else:
            self._logger.error('No Astra region provided. Region required to load data.')
            exit(1)

        if ks_name:
            self._astra_db_keyspace = ks_name
        else:
            self._logger.error('No Astra keyspace provided. Keyspace required to load data.')
            exit(1)

        if tbl_name:
            self._astra_db_table = tbl_name
        else:
            self._logger.error('No Astra table provided. Table required to load data.')
            exit(1)

        self._astra_api_endpoint = f'https://{self._astra_db_id}-{self._astra_db_region}.{self.astra_domain}'

    @staticmethod
    def _split_page_into_chunks(text_block, row_size_bytes):
        encoded_text_block = text_block.encode('utf-8')
        encoded_text_block_len = len(encoded_text_block)

        text_block_rtn = []
        start = 0
        while start < encoded_text_block_len:
            end = min(start + row_size_bytes, encoded_text_block_len)

            text_block_chunk = encoded_text_block[start:end].decode('utf-8', errors='ignore')
            text_block_rtn.append(text_block_chunk)

            start += len(text_block_chunk.encode('utf-8'))

        return text_block_rtn

    def load_markdown(self, markdown_index: MarkdownIndex, document_file_name: str, row_size_bytes: int) -> None:
        astra_client = ap.DataAPIClient(self._astra_token)
        astra_database = astra_client.get_database(self._astra_api_endpoint, keyspace=self._astra_db_keyspace)
        docs_collection = astra_database.get_collection(self._astra_db_table)

        document_title = markdown_index.get_title()
        markdown_dir = markdown_index.dir()

        for page_num, page_file in markdown_index:
            page_file_path = os.path.join(markdown_dir, page_file)
            self._logger.info(f'Preparing to load page: {page_file_path}')
            with open(page_file_path, 'r', encoding='utf-8') as page_file_h:
                md_chunks = self._split_page_into_chunks(page_file_h.read(), row_size_bytes)

            page_parts = len(md_chunks)
            if page_parts > 1:
                self._logger.info(f'Page {page_file} has been split into {page_parts} parts')

            for chunk_i, chunk_v in enumerate(md_chunks):
                if page_parts > 1:
                    page_value = f'{page_num}-{chunk_i + 1}'
                else:
                    page_value = page_num

                insert_result = docs_collection.insert_one({
                    'content': str(chunk_v),
                    '$vectorize': str(chunk_v),
                    'metadata': {
                        'source': document_file_name,
                        'title': document_title,
                        'page': page_value,
                        'language': 'English'
                    }
                })

                self._logger.info(insert_result)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s [%(name)s] %(asctime)s.%(msecs)03d - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    sys.exit(PdfContentProcessor.main_cli())
