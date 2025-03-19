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
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self.output_dir = ''
        self.base_name = ''
        self.source_dir = ''
        self.markdown_dir = ''

        self.source_index_path = ''
        self.source_index = None

        self.markdown_index_path = ''
        self.markdown_index = None

        self.parsing_token_file_path = ''
        self.parsing_instructions_path = ''
        self.parsing_retries = 3

        self.__dict__.update(parser_config_kwargs)

        print('DEBUG: PdfContentProcessor - REMOVE ME!')

    def _initialise_document_parser(self, abort_on_failure: bool) -> 'DocumentParser' or None:
        if os.path.exists(self.source_index_path):
            return None
        else:
            return DocumentParser(
                self.parsing_token_file_path,
                self.parsing_instructions_path,
                self.parsing_retries,
                abort_on_failure
            )

    def _prepare_dir(self, dir_path: str, delete_output: bool, delete_extn=None) -> None:
        if os.path.exists(dir_path):
            if delete_output:
                self._delete_dir_contents(dir_path, delete_extn=delete_extn)
        else:
            os.makedirs(dir_path, exist_ok=True)

    def _delete_dir_contents(self, dir_path: str, delete_extn=None) -> None:
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                if delete_extn:
                    if os.path.splitext(item_path)[1] in delete_extn:
                        os.unlink(item_path)
                else:
                    os.unlink(item_path)
            elif os.path.isdir(item_path):
                self._delete_dir_contents(item_path)
                os.rmdir(item_path)

    @staticmethod
    def main_cli() -> None:
        arg_parser = argparse.ArgumentParser(
            description='Processes a PDF file to extract its information using LlamaParse into markdown, and'
                        ' optionally load into an Astra Vector Database.'
        )
        arg_parser.add_argument('document_path', help='Path to the PDF file to parse.')
        arg_parser.add_argument(
            '--parsing_token_file_path',
            help='Path to the file containing the API token.'
        )
        arg_parser.add_argument(
            '--parsing_instructions_path',
            help='Path to text file containing the parsing instructions.'
        )
        arg_parser.add_argument(
            '--parsing_retries',
            help='Number of times to attempt to parse a document before giving up. Default is 3.',
            default=3
        )
        arg_parser.add_argument(
            '--astra_json_token_path',
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
            '-m',
            '--delete_markdown',
            help=f'Flag to delete all content and information contained in the \'markdown\' directory. The contents in'
                 f' the directory are deleted just before running the process that generates the markdown. If this'
                 f' flag is unspecified, and the {MarkdownIndex.file_name} is present in the \'markdown\' directory,'
                 f' the contents in the directory are preserved.',
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

        arg_parser.add_argument(
            '-s',
            '--delete_source',
            help=f'Flag to delete all content and information contained in the \'source\' directory. The contents in'
                 f' the directory are deleted just before running the process that extracts the document content. If'
                 f' this flag is unspecified, and the {SourceIndex.file_name} is present in the \'source\' directory,'
                 f' the contents in the directory are preserved.',
            action='store_true'
        )

        parsed_args = arg_parser.parse_args()

        document_parser_config = {
            'parsing_token_file_path': parsed_args.parsing_token_file_path,
            'parsing_instructions_path': parsed_args.parsing_instructions_path,
            'parsing_retries': parsed_args.parsing_retries,
        }

        document_loader_config = {
            'astra_json_token_path': parsed_args.astra_json_token_path,
            'astra_db_id': parsed_args.astra_db_id,
            'astra_db_region': parsed_args.astra_db_region,
            'astra_db_keyspace': parsed_args.astra_db_keyspace,
            'astra_db_table': parsed_args.astra_db_table,
            'load_database': parsed_args.load_database,
            'row_size_bytes': parsed_args.row_size_bytes
        }

        pdf_content_processor = PdfContentProcessor(**document_parser_config)
        pdf_content_processor.process_document(
            parsed_args.document_path,
            delete_source=parsed_args.delete_source,
            delete_markdown=parsed_args.delete_markdown,
            abort_on_failure=parsed_args.abort_on_failure,
            extract_images=parsed_args.extract_images,
            pages_per_split=parsed_args.pages_per_split,
        )

    def process_document(
            self,
            document_path: str,
            delete_source=False,
            delete_markdown=False,
            abort_on_failure=False,
            extract_images=False,
            pages_per_split=50,
            load_database=False
    ) -> None:
        self.logger.info(f"Using Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

        self.logger.info(f'Processing document {document_path}')
        self.output_dir = '.'.join(document_path.split('.')[:-1])

        self.logger.info(f'Generated artifacts will be stored in {self.output_dir}')
        self.base_name = self.output_dir.split('/')[-1]
        self.source_dir = os.path.join(self.output_dir, 'source')
        self.markdown_dir = os.path.join(self.output_dir, 'markdown')

        self.source_index_path = os.path.join(self.source_dir, SourceIndex.file_name)
        self.markdown_index_path = os.path.join(self.markdown_dir, MarkdownIndex.file_name)

        # Returns either a valid document parser object or None depending on the skip_llm_parsing flag and the
        # existence of the source index file. We want to do this now so we validate the parser arguments before we
        # start any directory setup.
        document_parser = self._initialise_document_parser(abort_on_failure)

        self._prepare_dir(self.source_dir, delete_source)
        self._prepare_dir(self.markdown_dir, delete_markdown, delete_extn=['.md', '.yaml'])

        if os.path.exists(self.source_index_path):
            self.source_index = SourceIndex.load(self.source_index_path)
        else:
            self.source_index = SourceIndex(base_name=self.base_name, source_dir=self.source_dir)
            self.logger.info('Splitting document into smaller documents')
            document_splitter = DocumentSplitter(self.source_index)
            document_splitter.split_document(document_path, self.source_dir, pages_per_split)

            document_parser.parse(self.source_index, extract_images)
            self.source_index.save()

        if os.path.exists(self.markdown_index_path):
            self.markdown_index = MarkdownIndex.load(self.markdown_index_path)
        else:
            self.markdown_index = MarkdownIndex(base_name=self.base_name, markdown_dir=self.markdown_dir)
            if not self.source_index:
                self.source_index = SourceIndex.load(self.source_index_path)

            # # TODO: Clean this up
            # markdown_page_files = self._process_raw_markdown(output_markdown_raw_dir_path, base_output_doc_name)
            # image_files = self._process_images(output_images_dir_path)
            #
            # logging.info('Document title: {}'.format(self.document_title))
            #
            # # The 'pages' key contains a list of dictionaries. Each dictionary contains information about a page in document
            # # path to the processed markdown information for a page and the document page number
            # # that the information came from:
            # #
            # #      'pages': [{'page': <page_num>, 'path': <output_page_path>}, ...]
            # return {'title': self.document_title, 'pages': markdown_page_files, 'images': image_files}

        self.logger.info('Processing complete!')

        exit()



        if parsed_args.load_database:
            astra_loader = DocumentLoader(
                parsed_args.astra_json_token_path,
                parsed_args.astra_db_id,
                parsed_args.astra_db_region,
                parsed_args.maximum_row_size_bytes
            )

            astra_loader.load_document_data(
                parsed_args.astra_db_keyspace,
                parsed_args.astra_db_table,
                base_output_name,
                document_content['title'],
                document_content['pages']
            )


    @staticmethod
    def write_markdown_page(file_path, markdown_output):
        with open(file_path, 'w', encoding='utf-8') as file_out_h:
            file_out_h.write(markdown_output)
            file_out_h.write('\n\n')

    @staticmethod
    def append_markdown_page(file_path, markdown_output):
        with open(file_path, 'a', encoding='utf-8') as file_out_h:
            file_out_h.write(markdown_output)
            file_out_h.write('\n\n')

    @staticmethod
    def process_images(images_dir_path):
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

    def __init__(self, base_name=None, source_dir=None, document_extn='.pdf'):
        super().__init__(base_name, source_dir)
        self.document_extn = document_extn

        self._source_ranges = []
        self._source_range_info = {}

    def __iter__(self):
        for range_name in self._source_ranges:
            yield range_name, self._source_range_info[range_name]

    def append_page_range(self, start_page: int, end_page: int) -> (str, str):
        range_name = f'pages_{start_page}-{end_page}'
        file_name = f'{self._base_name}_{range_name}{self.document_extn}'
        self._source_ranges.append(range_name)
        self._source_range_info[range_name] = {
            'file': file_name,
            'range': (start_page, end_page)
        }

        return range_name, file_name

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
        with open(index_path, 'w', encoding='utf-8') as index_file_h:
            yaml.dump(index_data, index_file_h)


class MarkdownIndex(OutputIndex):
    file_name = 'markdown_index.yaml'

    def __init__(self, base_name=None, markdown_dir=None):
        super().__init__(base_name, markdown_dir)

        self.markdown_pages = []

    def append_page(self, page_num: int, markdown_file: str) -> None:
        self.markdown_pages.append({
            'page': page_num,
            'file': markdown_file
        })

    # YAML file is in the format
    # ---
    # markdown:
    #   base_name: <basename>
    #   output_dir: <output_directory>
    #   pages:
    #     - page: <page_num>
    #       file: <split_filename>
    @staticmethod
    def load(index_file: str) -> OutputIndex:
        with open(index_file, 'r', encoding='utf-8') as index_file_h:
            index_data = yaml.safe_load(index_file_h)

            markdown_info = index_data['markdown']
            markdown_index = MarkdownIndex(
                base_name=markdown_info['base_name'],
                markdown_dir=markdown_info['output_dir']
            )

            for page_info in markdown_info['pages']:
                markdown_index.append_page(page_info['page'], page_info['file'])

            return markdown_index

    def save_index(self) -> None:
        index_data = {
            'markdown': {
                'base_name': self._base_name,
                'output_dir': self._output_dir,
                'pages': [] + self.markdown_pages
            }
        }

        index_path = os.path.join(self._output_dir, MarkdownIndex.file_name)
        with open(index_path, 'w', encoding='utf-8') as index_file_h:
            yaml.dump(index_data, index_file_h)

class DocumentSplitter:
    def __init__(self, source_index: SourceIndex, logger=None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
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
                self.logger.warning(f'Skipping {output_doc_path} as it already exists.')
                continue

            pdf_writer = pp2.PdfWriter()
            for page_num in range(start_page, end_page):
                pdf_writer.add_page(pdf_reader.pages[page_num])

            with open(output_doc_path, "wb") as output_pdf:
                pdf_writer.write(output_pdf)

            self.logger.info(f'Created {output_doc_path} containing pages {start_page + 1} to {end_page}.')


class DocumentParser:
    def __init__(self, token_file: str, instructions_file: str, retry_count=3, abort_on_failure=False, logger=None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.llama_parse_parser = None
        self.parsing_retries = int(retry_count)
        self.abort_on_failure = abort_on_failure

        self.page_pattern = r'^\[(title|contents|information|blank)\]: #$'
        self.document_title = ''

        self._initialise_parser(token_file, instructions_file)

    def _initialise_parser(self, token_file: str, instructions_file: str) -> None:
        llama_parse_api_key = ''
        llama_parse_instructions = ''

        if token_file:
            with open(token_file, 'r') as file_h:
                llama_parse_api_key = file_h.read().strip()
        else:
            self.logger.error('No LlamaParse token provided. Token required to parse data.')
            exit(1)

        if instructions_file:
            with open(instructions_file, 'r') as file_h:
                llama_parse_instructions = ''.join(line for line in file_h if not line.startswith('#'))
        else:
            self.logger.warning(
                'No parsing instructions provided. This will likely result in the document being parsed incorrectly.')
            while True:
                response = input('Do you want to continue? (Y/N): ').strip().lower()
                if response:
                    if response[0] == 'y':
                        break
                    elif response[0] == 'n':
                        exit(0)

                print("Invalid input. Please enter 'Y' to continue, or 'N' to abort.")

        self.llama_parse_parser = lp.LlamaParse(
            api_key=llama_parse_api_key,
            result_type=lp.ResultType.MD,
            parsing_instruction=llama_parse_instructions,
            verbose=True,
            invalidate_cache=True,
            do_not_cache=True,
        )

    def _generate_raw_output(self, document_path: str, markdown_raw_dir: str, base_name: str, images_dir: str) -> None:
        document_content = self._extract_document_content_with_llm(document_path)

        if not document_content:
            return

        self.logger.info(document_content[0]['job_metadata'])

        for extracted_content in document_content[0]['pages']:
            extracted_markdown = self._remove_markdown_markers_and_whitespace(extracted_content['md'])
            markdown_raw_file_name = '{}_page_{}.md'.format(base_name, str(extracted_content['page']).zfill(3))
            markdown_raw_file_path = os.path.join(markdown_raw_dir, markdown_raw_file_name)

            PdfContentProcessor.write_markdown_page(markdown_raw_file_path, extracted_markdown)

            if images_dir:
                self.llama_parse_parser.get_images(
                    document_content,
                    download_path=images_dir
                )

    def _extract_document_content_with_llm(self, document_path: str) -> list or None:
        self.logger.info(f'Parsing document {document_path}')
        retry_count = 0
        job_response = []
        while retry_count < self.parsing_retries:
            # Attempt to parse the document using the LlamaParse API
            job_response = self.llama_parse_parser.get_json_result(document_path)

            if len(job_response) > 0:
                break

            self.logger.info('Failed to parse document. Retrying...')
            retry_count += 1

        if len(job_response) < 1:
            if self.abort_on_failure:
                self.logger.error('Failed to parse document after {} retries. Aborting.'.format(self.parsing_retries))
                exit(1)
            else:
                self.logger.warning('Failed to parse document after {} retries. Skipping.'.format(self.parsing_retries))
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
    def parse(self, source_index: SourceIndex, extract_images: bool) -> None:
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
                source_part_raw_dir,
                source_index.base_name(),
                source_part_image_dir
            )


class RawContentParser:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    # Parsing of the raw markdown pages written to disk; this will piece together the title, split and group page
    # information according to how the original document was structured, and determine the page in the document
    # information was extracted from
    def parse(self, source_index: SourceIndex, markdown_index: MarkdownIndex) -> None:
        for source_range in source_index:
            source_range_name, source_range_info = source_range
            source_range_dir = os.path.join(source_index.dir(), source_range_name, 'raw')

            self._process_raw_markdown(
                source_range_dir,
                source_index.base_name(),
                markdown_index
            )

    def _process_raw_markdown(self, raw_markdown_dir_path, base_output_doc_name):
        markdown_page_files = []
        markdown_raw_page_files = [
            f for f in os.listdir(raw_markdown_dir_path) if os.path.isfile(os.path.join(raw_markdown_dir_path, f))
        ]
        markdown_raw_page_files.sort()

        page_num = 0
        page_type = None
        for raw_markdown_file in markdown_raw_page_files:
            raw_markdown_file_path = os.path.join(raw_markdown_dir_path, raw_markdown_file)

            # Open the file with UTF-8 encoding
            with open(raw_markdown_file_path, 'r',encoding='utf-8') as raw_markdown_file_h:
                raw_markdown = raw_markdown_file_h.read()

                logging.info('Parsing markdown content for page {}'.format(raw_markdown_file_path))
                extracted_markdown_pages = self._parse_markdown(raw_markdown, page_num, page_type)

                for markdown_page in extracted_markdown_pages:
                    # Store for the current page the number and type. The previous raw markdown page for this nested
                    # loop iteration may flow on to the current raw markdown page. Storing the page number and type
                    # will allow us to correctly set the page properties for this raw page that is parsed.
                    page_num = markdown_page['page']
                    page_type = markdown_page['type']
                    output_page_path = '{}_page_{}.md'.format(base_output_doc_name, str(page_num).zfill(3))

                    if os.path.exists(output_page_path):
                        logging.info('Appending markdown page {}'.format(output_page_path))
                        self._append_markdown_page(output_page_path, markdown_page['md'])
                    else:
                        logging.info('Writing markdown page {}'.format(output_page_path))
                        self._write_markdown_page(output_page_path, markdown_page['md'])

                        # Only add to the page files list if we have a new page. Otherwise, we will have duplicate
                        # page entries, if we add to the page file list when we are appending markdown to an existing
                        # page.
                        markdown_page_files.append({'page': page_num, 'path': output_page_path})

        return markdown_page_files

    def _parse_markdown(self, extracted_page, previous_page_number=0, previous_page_type=None):
        # Keep tract of the previous page number so we know where we are up to in the document parsing and to handle
        # possible case where the content was split over separate raw markdown pages
        extracted_page_lines = extracted_page.split('\n')
        markdown_pages = []
        markdown_page_line_buffer = []
        markdown_page_number = previous_page_number
        markdown_page_type = previous_page_type

        # The parsing of the markdown relies heavily on the instructions passed to LlamaParse. For the parsing to work
        # correctly, LlamaParse must be instructed to place the page type at the very start of the markdown for that
        # page.
        for line in extracted_page_lines:
            page_type_match = re.match(self.page_pattern, line)
            if page_type_match:
                # Depending on the content size, LlamaParse will sometime combine the markdown for multiple pages into
                # one page. We want to split them back into individual pages so the source of the information can be
                # easily traced back. If we hit a new page match while we are still processing the extracted page, we
                # need to save the content in the buffer before we start processing the new page.
                markdown_pages += self._parse_markdown_construct_page(
                    markdown_page_number,
                    markdown_page_type,
                    markdown_page_line_buffer
                )

                # Increment page number and assign new page type
                markdown_page_number += 1
                markdown_page_type = page_type_match.groups()[0]
                markdown_page_line_buffer = []

                continue
            # End page_type_match handling

            # Depending on the current page type, we will either save the content to the buffer or do nothing
            if markdown_page_type == 'information':
                markdown_page_line_buffer.append(line)
            elif markdown_page_type == 'title':
                markdown_page_line_buffer.append(line.strip())
        # End extracted_page_lines for loop

        # Check if there is any lines in the buffer that need to be saved before we exit
        markdown_pages += self._parse_markdown_construct_page(
            markdown_page_number,
            markdown_page_type,
            markdown_page_line_buffer
        )

        return markdown_pages

    def _parse_markdown_construct_page(self, page_number, page_type, page_buffer):
        constructed_pages = []
        if page_type == 'information':
            if page_buffer:
                constructed_pages.append({
                    'page': page_number,
                    'md': '\n'.join(page_buffer),
                    'type': page_type
                })
        elif page_type == 'title':
            if not self.document_title:
                self.document_title = ' '.join(page_buffer).title().strip()
                logging.info('Extracted document title: {}'.format(self.document_title))
        elif page_type == 'contents' or page_type == 'blank':
            logging.info('Skipping page {} as it is a {} page'.format(str(page_number).zfill(3), page_type))

        return constructed_pages


class DocumentLoader:
    def __init__(self, astra_token_file, astra_db_id, astra_region, max_row_size_bytes=8000):
        self.astra_token = None
        self.astra_db_id = None
        self.astra_region = None
        self.max_row_size_bytes = int(max_row_size_bytes)
        self.astra_domain = 'apps.astra.datastax.com'

        if astra_token_file:
            with open(astra_token_file, 'r') as file_h:
                astra_token = json.loads(file_h.read())
                self.astra_token = astra_token['token']
        else:
            logging.error('No Astra token provided. Token required to load data.')
            exit(1)

        if astra_db_id:
            self.astra_db_id = astra_db_id
        else:
            logging.error('No Astra database ID provided. Database ID required to load data.')
            exit(1)

        if astra_region:
            self.astra_region = astra_region
        else:
            logging.error('No Astra region provided. Region required to load data.')
            exit(1)

        self.astra_api_endpoint = 'https://{}-{}.{}'.format(
            self.astra_db_id,
            self.astra_region,
            self.astra_domain
        )


    def load_document_data(self, keyspace_name, table_name, document_name, document_title, document_pages):
        astra_client = ap.DataAPIClient(self.astra_token)
        astra_database = astra_client.get_database(self.astra_api_endpoint, keyspace=keyspace_name)
        docs_collection = astra_database.get_collection(table_name)

        # [{'page': page_num, 'path': output_page_path}, ...]
        collection_pages = []
        for page_info in document_pages:
            logging.info('Preparing to load page: {}'.format(page_info['path']))
            with open(page_info['path'], 'r', encoding='utf-8') as page_file_h:
                md_chunks = self._split_page_into_chunks(page_file_h.read())

            page_parts = len(md_chunks)
            if page_parts > 1:
                logging.info('Page {} has been split into {} parts'.format(page_info['page'], page_parts))

            for chunk_i, chunk_v in enumerate(md_chunks):
                if page_parts > 1:
                    page_value = '{}-{}'.format(page_info['page'], chunk_i + 1)
                else:
                    page_value = page_info['page']

                collection_pages.append({
                    'content': str(chunk_v),
                    '$vectorize': str(chunk_v),
                    'metadata': {
                        'source': document_name,
                        'title': document_title,
                        'page': page_value,
                        'language': 'English'
                    }
                })

        insert_result = docs_collection.insert_many(collection_pages)
        print(insert_result)

    def _split_page_into_chunks(self, text_block):
        encoded_text_block = text_block.encode('utf-8')
        encoded_text_block_len = len(encoded_text_block)

        text_block_rtn = []
        start = 0
        while start < encoded_text_block_len:
            end = min(start + self.max_row_size_bytes, encoded_text_block_len)

            text_block_chunk = encoded_text_block[start:end].decode('utf-8', errors='ignore')
            text_block_rtn.append(text_block_chunk)

            start += len(text_block_chunk.encode('utf-8'))

        return text_block_rtn


if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s [%(name)s] %(asctime)s.%(msecs)03d - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    sys.exit(PdfContentProcessor.main_cli())

    # arg_parser = argparse.ArgumentParser(description='Processes a PDF file to extract its information using LlamaParse into markdown, and optionally load into an Astra Vector Database.')
    # arg_parser.add_argument('document_path', help='Path to the PDF file to parse.')
    # arg_parser.add_argument('--parsing_token_file_path', help='Path to the file containing the API token.')
    # arg_parser.add_argument('--parsing_instructions_path', help='Path to text file containing the parsing instructions.')
    # arg_parser.add_argument('--parsing_retries', help='Number of times to attempt to parse a document before giving up. Default is 3.', default=3)
    # arg_parser.add_argument('--astra_json_token_path', help='Path to the file containing the Astra API JSON token.')
    # arg_parser.add_argument('--astra_db_id', help='Astra database ID.')
    # arg_parser.add_argument('--astra_db_region', help='Astra region.')
    # arg_parser.add_argument('--astra_db_keyspace', help='Astra keyspace.')
    # arg_parser.add_argument('--astra_db_table', help='Astra table.')
    # arg_parser.add_argument('-a', '--abort_on_failure', help='Flag to abort processing of document if parsing fails after retrying.', action='store_true')
    # arg_parser.add_argument('-i', '--extract_images', help='Flag to extract images from the document.', action='store_true')
    # arg_parser.add_argument('-l', '--load_database', help='Flag to load the document into Astra.', action='store_true')
    # arg_parser.add_argument('-s', '--skip_llm_parsing', help='Flag to skip LlamaParse extraction, and parse existing raw output.', action='store_true')
    # arg_parser.add_argument('-r', '--maximum_row_size_bytes', help='Maximum row size in bytes used to generate embeddings. Default is 8000.', default=8000)
    #
    # parsed_args = arg_parser.parse_args()
    #
    # print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    #
    # logging.basicConfig(
    #     format='%(levelname)s [%(name)s] %(asctime)s - %(message)s',
    #     level=logging.INFO,
    #     datefmt='%Y-%m-%d %H:%M:%S.%f'
    # )
    #
    # logging.info('Processing document {}'.format(parsed_args.document_path))
    # base_output_dir = '.'.join(parsed_args.document_path.split('.')[:-1])
    # logging.info('Generated artifacts will be stored in {}'.format(base_output_dir))
    # base_output_name = base_output_dir.split('/')[-1]
    #
    # source_output_dir = os.path.join(base_output_dir, 'source')
    # markdown_raw_output_dir = os.path.join(base_output_dir, 'markdown_raw')
    # markdown_output_dir = os.path.join(base_output_dir, 'markdown')
    # images_output_dir = None
    #
    # os.makedirs(source_output_dir, exist_ok=True)
    # os.makedirs(markdown_raw_output_dir, exist_ok=True)
    #
    # try:
    #     os.makedirs(markdown_output_dir, exist_ok=False)
    # except FileExistsError:
    #     for markdown_file_name in os.listdir(markdown_output_dir):
    #         markdown_file_path = os.path.join(markdown_output_dir, markdown_file_name)
    #
    #         # Check if it is a file before deleting
    #         if os.path.isfile(markdown_file_path) and markdown_file_name.endswith('.md'):
    #             os.remove(markdown_file_path)
    #
    # if parsed_args.extract_images:
    #     images_output_dir = os.path.join(base_output_dir, 'images')
    #     os.makedirs(images_output_dir, exist_ok=True)
    #
    #
    # if parsed_args.skip_llm_parsing:
    #     split_document_path_list = []
    # else:
    #     logging.info('Splitting document into smaller documents')
    #     document_splitter = DocumentSplitter()
    #     split_document_path_list = document_splitter.split_document(parsed_args.document_path, source_output_dir)
    #
    # document_parser = DocumentParser(
    #     parsed_args.parsing_token_file_path,
    #     parsed_args.parsing_instructions_path,
    #     parsed_args.parsing_retries,
    #     parsed_args.abort_on_failure,
    #     parsed_args.skip_llm_parsing
    # )
    #
    # # The parse_document method will return a dict with the following information. The 'pages' key will contain all
    # # the extracted page content and the page number the content was extracted from.
    # #
    # # document_content = {
    # #   'title': self.document_title,
    # #   'pages': [{'page': page_num, 'path': output_page_path}, ...],
    # #   'images': image_files
    # # }
    # document_content = document_parser.parse_documents(
    #     split_document_path_list,
    #     markdown_output_dir,
    #     markdown_raw_output_dir,
    #     base_output_name,
    #     images_output_dir
    # )
    #
    # if parsed_args.load_database:
    #     astra_loader = DocumentLoader(
    #         parsed_args.astra_json_token_path,
    #         parsed_args.astra_db_id,
    #         parsed_args.astra_db_region,
    #         parsed_args.maximum_row_size_bytes
    #     )
    #
    #     astra_loader.load_document_data(
    #         parsed_args.astra_db_keyspace,
    #         parsed_args.astra_db_table,
    #         base_output_name,
    #         document_content['title'],
    #         document_content['pages']
    #     )
    #
    # logging.info('Processing complete!')
