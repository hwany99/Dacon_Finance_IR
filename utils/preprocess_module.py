import fitz  # PyMuPDF
from extract_module import daconCustomExtractor
import pandas as pd
from tqdm import tqdm
import os
import unicodedata
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdf(file_path):
    # 1. pdf별 page 추출
    doc = fitz.open(file_path)
    outpt_dir = "/home/a2024712006/dacon/extract_image"
    chunk_list = []
    for page_number in range(doc.page_count):
        page = doc.load_page(page_number)
        
        #2. page별 테이블 추출 
        tables = page.find_tables()
        raw_text_list = []
        for table in tables:
            # 테이블을 감싸는 영역 계산
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            
            for cell in table.cells:
                x0, y0, x1, y1 = cell[:4]  # 셀 좌표 추출
                min_x = min(min_x, x0)
                min_y = min(min_y, y0)
                max_x = max(max_x, x1)
                max_y = max(max_y, y1)
            # 1) 발견된 테이블 영역 
            table_rect = fitz.Rect(min_x, min_y, max_x, max_y)
            # 2) 발견된 테이블 영역의 테이블 형식 텍스트
            table_text = "\n"
            for row in table.extract():
                table_text += str(row)
                table_text += "\n"
            # 3) 발견된 테이블 영역의 날 것 텍스트 
            clipped_text = page.get_text("text", clip=table_rect)
            
            # 4) 날 것 텍스트 => 테이블 형식 텍스트로 변환
            raw_text_list.append((clipped_text, table_text))
            
        # 2. page별 이미지 추출        
        extractor = daconCustomExtractor(page)
        print(file_path)
        print(f"Extracting text from page {page_number + 1}...")        
        bboxes = extractor.detect_svg_contours(page_number+1, output_dir=outpt_dir, min_svg_gap_dx=25.0, min_svg_gap_dy=25.0, min_w=2.0, min_h=2.0)

        # 텍스트를 chunk로 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=32
        )    
        # 3. 이미지별 텍스트 추출
        for i, bbox in enumerate(bboxes):
            x0, y0, x1, y1 = bbox 
            
            full_text = page.get_text("text", clip=fitz.Rect(x0, y0, x1, y1))
            for clipped_text, table_text in raw_text_list:
                if clipped_text in full_text:
                    full_text = full_text.replace(clipped_text, table_text)
            
            
            chunk_temp = splitter.split_text(full_text)
            chunk_list.extend(chunk_temp)
    return chunk_list

def normalize_path(path):
    """경로 유니코드 정규화"""
    return unicodedata.normalize('NFC', path)

def normalize_string(s):
    """유니코드 정규화"""
    return unicodedata.normalize('NFC', s)

def process_pdfs_from_dataframe(df, base_directory):

    unique_paths = df['Source_path'].unique()

    data_frames = {}
    for path in tqdm(unique_paths, desc="Processing PDFs"):
        # 경로 정규화 및 절대 경로 생성
        normalized_path = normalize_path(path)

        full_path = os.path.normpath(os.path.join(base_directory, normalized_path.lstrip('./'))) if not os.path.isabs(normalized_path) else normalized_path

        pdf_title = os.path.splitext(os.path.basename(full_path))[0]
        print(pdf_title)
        # PDF 처리 및 벡터 DB 생성
        chunks = process_pdf(full_path)
        data_frames[pdf_title] = chunks

    return data_frames

def key_index(key):
    dict_map = {
        '중소벤처기업부_혁신창업사업화자금(융자)': "index_0",
        '보건복지부_부모급여(영아수당) 지원': "index_1",
        '보건복지부_노인장기요양보험 사업운영': "index_2",
        '산업통상자원부_에너지바우처': "index_3",
        '국토교통부_행복주택출자': "index_4",
        '「FIS 이슈 & 포커스」 22-4호 《중앙-지방 간 재정조정제도》': "index_5",
        '「FIS 이슈 & 포커스」 23-2호 《핵심재정사업 성과관리》': "index_6",
        '「FIS 이슈&포커스」 22-2호 《재정성과관리제도》': "index_7",
        '「FIS 이슈 & 포커스」(신규) 통권 제1호 《우발부채》': "index_8"
    }
    return dict_map[key]