#!/usr/bin/env python

import argparse
import json
import os
import re

import astrapy as ap
import llama_parse as lp
import llama_index.core.schema as llcs
import PyPDF2 as pp2


class DocumentSplitter:
    @staticmethod
    def split_document(input_document_path, output_dir_path, pages_per_split=50):
        document_base_name = input_document_path.split('/')[-1].split('.')[0]
        pdf_reader = pp2.PdfReader(input_document_path)

        num_pages_total = len(pdf_reader.pages)
        num_documents = (num_pages_total + pages_per_split - 1) // pages_per_split
        split_documents = []

        for doc_index in range(num_documents):
            start_page = doc_index * pages_per_split
            end_page = min(start_page + pages_per_split, num_pages_total)
            output_doc_name = '{}_pages_{}-{}.pdf'.format(
                document_base_name,
                str(start_page + 1).zfill(3),
                str(end_page).zfill(3)
            )
            output_doc_path = os.path.join(output_dir_path, output_doc_name)
            split_documents.append(output_doc_path)

            if os.path.exists(output_doc_path):
                print('Skipping {} as it already exists.'.format(output_doc_path))
                continue

            pdf_writer = pp2.PdfWriter()
            for page_num in range(start_page, end_page):
                pdf_writer.add_page(pdf_reader.pages[page_num])

            with open(output_doc_path, "wb") as output_pdf:
                pdf_writer.write(output_pdf)

            print('Created {} containing pages {} to {}.'.format(output_doc_path, start_page + 1, end_page))

        return split_documents


class DocumentParser:
    def __init__(self, token_file, instructions_file, retry_count=3, abort_on_failure=False, skip_llm_parsing=False):
        self.skip_llm_parsing = skip_llm_parsing
        self.llama_parse_api_key = None
        self.llama_parse_instructions = None
        self.llama_parse_parser = None
        self.parsing_retries = int(retry_count)
        self.abort_on_failure = abort_on_failure

        self.page_pattern = r'^\[(title|contents|information|blank)\]: #$'
        self.document_title = ''

        if self.skip_llm_parsing:
            return

        self._initialise_parser(token_file, instructions_file)

    def parse_documents(
            self,
            input_document_paths,
            output_markdown_dir_path,
            output_markdown_raw_dir_path,
            output_name,
            output_images_dir_path=None
    ):
        # There are two main steps to parse the document:
        #  1. Content extraction using LlamaParse; this will extract the document information into raw markdown pages
        #     which are then written to disk
        #  2. Parsing of the raw markdown pages written to disk; this will piece together the title, split and group
        #     page information according to how the original document was structured, and determine the page in the
        #     document information was extracted from

        base_output_doc_name = '{}/{}'.format(output_markdown_dir_path, output_name)

        if self.llama_parse_parser:
            self._generate_raw_output(
                input_document_paths,
                output_markdown_raw_dir_path,
                output_name,
                output_images_dir_path
            )

        markdown_page_files = self._process_raw_markdown(output_markdown_raw_dir_path, base_output_doc_name)
        image_files = self._process_images(output_images_dir_path)

        print('Document title set to "{}"'.format(self.document_title))

        # The 'pages' key contains a list of dictionaries. Each dictionary contains information about a page in document
        # path to the processed markdown information for a page and the document page number
        # that the information came from:
        #
        #      'pages': [{'page': <page_num>, 'path': <output_page_path>}, ...]
        return {'title': self.document_title, 'pages': markdown_page_files, 'images': image_files}

    def _initialise_parser(self, token_file, instructions_file):
        if token_file:
            with open(token_file, 'r') as file_h:
                self.llama_parse_api_key = file_h.read().strip()
        else:
            print('ERROR: No LlamaParse token provided. Token required to parse data.')
            exit(1)

        if instructions_file:
            with open(instructions_file, 'r') as file_h:
                self.llama_parse_instructions = ''.join(line for line in file_h if not line.startswith('#'))
        else:
            while True:
                response = input(
                    'WARNING: No parsing instructions provided. This will likely result in the document being'
                    ' parsed incorrectly. Do you want to continue? (Y/N): '
                ).strip().lower()
                if response:
                    if response[0] == 'y':
                        break
                    elif response[0] == 'n':
                        exit(0)

                print("Invalid input. Please enter 'Y' to continue, or 'N' to abort.")

        self.llama_parse_parser = lp.LlamaParse(
            api_key=self.llama_parse_api_key,
            result_type='markdown',
            parsing_instruction=self.llama_parse_instructions,
            verbose=True,
            invalidate_cache=True,
            do_not_cache=True,
            continuous_mode=True,
        )

    def _generate_raw_output(self, input_document_paths, output_markdown_raw_dir_path, output_name, output_images_dir_path):
        for document_path in input_document_paths:
            document_content = self._extract_document_content_with_llm(document_path)
            if not document_content:
                continue

            print(document_content[0]['job_metadata'])

            for extracted_content in document_content[0]['pages']:
                extracted_markdown = self._remove_markdown_markers_and_whitespace(extracted_content['md'])
                self._write_markdown_page(
                    '{}/{}_page_{}.md'.format(
                        output_markdown_raw_dir_path,
                        output_name,
                        str(extracted_content['page']).zfill(3)
                    ),
                    extracted_markdown
                )

            if output_images_dir_path:
                self.llama_parse_parser.get_images(
                    document_content,
                    download_path=output_images_dir_path
                )

    def _extract_document_content_with_llm(self, document_path):
        print('Parsing document {}'.format(document_path))
        retry_count = 0
        while retry_count < self.parsing_retries:
            # Attempt to parse the document using the LlamaParse API
            job_response = self.llama_parse_parser.get_json_result(document_path)

            if len(job_response) > 0:
                break

            'Failed to parse document. Retrying...'
            retry_count += 1

        if len(job_response) < 1:
            if self.abort_on_failure:
                print('Failed to parse document after {} retries. Aborting.'.format(self.parsing_retries))
                exit(1)
            else:
                print('Failed to parse document after {} retries. Skipping.'.format(self.parsing_retries))
                return None

        return job_response

    def _remove_markdown_markers_and_whitespace(self, page_markdown):
        processed_page = page_markdown

        if processed_page.startswith('```markdown'):
            processed_page = processed_page[len('```markdown'):]

        if processed_page.endswith('```'):
            processed_page = processed_page[:-len('```')]

        return processed_page.strip()

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
            with open(raw_markdown_file_path, 'r') as raw_markdown_file_h:
                raw_markdown = raw_markdown_file_h.read()

                print('Parsing markdown content for page {}'.format(raw_markdown_file_path))
                extracted_markdown_pages = self._parse_markdown(raw_markdown, page_num, page_type)
                for markdown_page in extracted_markdown_pages:
                    # Store for the current page the number and type. The previous raw markdown page for this nested
                    # loop iteration may flow on to the current raw markdown page. Storing the page number and type
                    # will allow us to correctly set the page properties for this raw page that is parsed.
                    page_num = markdown_page['page']
                    page_type = markdown_page['type']
                    output_page_path = '{}_page_{}.md'.format(base_output_doc_name, str(page_num).zfill(3))

                    if os.path.exists(output_page_path):
                        print('Appending markdown page {}'.format(output_page_path))
                        self._append_markdown_page(output_page_path, markdown_page['md'])
                    else:
                        print('Writing markdown page {}'.format(output_page_path))
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
                print('Extracting document title from page {}'.format(str(page_number).zfill(3)))
        elif page_type == 'contents' or page_type == 'blank':
            print('Skipping page {} as it is a {} page'.format(str(page_number).zfill(3), page_type))

        return constructed_pages

    def _write_markdown_page(self, file_path, markdown_output):
        with open(file_path, 'w') as file_out_h:
            file_out_h.write(markdown_output)
            file_out_h.write('\n\n')

    def _append_markdown_page(self, file_path, markdown_output):
        with open(file_path, 'a') as file_out_h:
            file_out_h.write(markdown_output)
            file_out_h.write('\n\n')

    def _write_image_files(self, file_image_dir, parser_result):
        image_info_list = self.llama_parse_parser.get_images(parser_result, download_path=file_image_dir)
        return [llcs.ImageDocument(image_path=image_info['path']) for image_info in image_info_list]

    def _process_images(self, images_dir_path):
        image_files = []
        if images_dir_path:
            image_files = [
                f for f in os.listdir(images_dir_path) if os.path.isfile(os.path.join(images_dir_path, f))
            ]
            image_files.sort()

        return image_files


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
            print('ERROR: No Astra token provided. Token required to load data.')
            exit(1)

        if astra_db_id:
            self.astra_db_id = astra_db_id
        else:
            print('ERROR: No Astra database ID provided. Database ID required to load data.')
            exit(1)

        if astra_region:
            self.astra_region = astra_region
        else:
            print('ERROR: No Astra region provided. Region required to load data.')
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
            print('Preparing to load page: {}'.format(page_info['path']))
            with open(page_info['path'], 'r') as page_file_h:
                md_chunks = self._split_page_into_chunks(page_file_h.read())

            page_parts = len(md_chunks)
            if page_parts > 1:
                print('Page {} has been split into {} parts'.format(page_info['page'], page_parts))

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
    arg_parser = argparse.ArgumentParser(description='Processes a PDF file to extract its information using LlamaParse into markdown, and optionally load into an Astra Vector Database.')
    arg_parser.add_argument('document_path', help='Path to the PDF file to parse.')
    arg_parser.add_argument('--document_title', help='Title of the document.', default='')
    arg_parser.add_argument('--parsing_token_file_path', help='Path to the file containing the API token.')
    arg_parser.add_argument('--parsing_instructions_path', help='Path to text file containing the parsing instructions.')
    arg_parser.add_argument('--parsing_retries', help='Number of times to attempt to parse a document before giving up. Default is 3.', default=3)
    arg_parser.add_argument('--astra_json_token_path', help='Path to the file containing the Astra API JSON token.')
    arg_parser.add_argument('--astra_db_id', help='Astra database ID.')
    arg_parser.add_argument('--astra_db_region', help='Astra region.')
    arg_parser.add_argument('--astra_db_keyspace', help='Astra keyspace.')
    arg_parser.add_argument('--astra_db_table', help='Astra table.')
    arg_parser.add_argument('-a', '--abort_on_failure', help='Flag to abort processing of document if parsing fails after retrying.', action='store_true')
    arg_parser.add_argument('-i', '--extract_images', help='Flag to extract images from the document.', action='store_true')
    arg_parser.add_argument('-l', '--load_database', help='Flag to load the document into Astra.', action='store_true')
    arg_parser.add_argument('-s', '--skip_llm_parsing', help='Flag to skip LlamaParse extraction, and parse existing raw output.', action='store_true')
    arg_parser.add_argument('-r', '--maximum_row_size_bytes', help='Maximum row size in bytes used to generate embeddings. Default is 8000.', default=8000)

    parsed_args = arg_parser.parse_args()

    print('Processing document {}'.format(parsed_args.document_path))
    base_output_dir = '.'.join(parsed_args.document_path.split('.')[:-1])
    print('Generated artifacts will be stored in {}'.format(base_output_dir))
    base_output_name = base_output_dir.split('/')[-1]

    source_output_dir = os.path.join(base_output_dir, 'source')
    markdown_raw_output_dir = os.path.join(base_output_dir, 'markdown_raw')
    markdown_output_dir = os.path.join(base_output_dir, 'markdown')
    images_output_dir = None

    os.makedirs(source_output_dir, exist_ok=True)
    os.makedirs(markdown_raw_output_dir, exist_ok=True)

    try:
        os.makedirs(markdown_output_dir, exist_ok=False)
    except FileExistsError:
        for markdown_file_name in os.listdir(markdown_output_dir):
            markdown_file_path = os.path.join(markdown_output_dir, markdown_file_name)

            # Check if it is a file before deleting
            if os.path.isfile(markdown_file_path) and markdown_file_name.endswith('.md'):
                os.remove(markdown_file_path)

    if parsed_args.extract_images:
        images_output_dir = os.path.join(base_output_dir, 'images')
        os.makedirs(images_output_dir, exist_ok=True)


    if parsed_args.skip_llm_parsing:
        split_document_path_list = []
    else:
        print('Splitting document into smaller documents')
        document_splitter = DocumentSplitter()
        split_document_path_list = document_splitter.split_document(parsed_args.document_path, source_output_dir)

    document_parser = DocumentParser(
        parsed_args.parsing_token_file_path,
        parsed_args.parsing_instructions_path,
        parsed_args.parsing_retries,
        parsed_args.abort_on_failure,
        parsed_args.skip_llm_parsing
    )

    # The parse_document method will return a dict with the following information. The 'pages' key will contain all
    # the extracted page content and the page number the content was extracted from.
    #
    # document_content = {
    #   'title': self.document_title,
    #   'pages': [{'page': page_num, 'path': output_page_path}, ...],
    #   'images': image_files
    # }
    document_content = document_parser.parse_documents(
        split_document_path_list,
        markdown_output_dir,
        markdown_raw_output_dir,
        base_output_name,
        images_output_dir
    )

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

    print('Processing complete!')
