from extract import daconCustomExtractor
from langchain.schema import Document
import fitz  # PyMuPDF

def process_pdf(file_path):
    # 1. pdf별 page 추출
    doc = fitz.open(file_path)
    chunk_temp = []
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
        bboxes = extractor.detect_svg_contours(page_number+1, min_svg_gap_dx=25.0, min_svg_gap_dy=25.0, min_w=2.0, min_h=2.0)

                
        # 3. 이미지별 텍스트 추출
        for i, bbox in enumerate(bboxes):
            x0, y0, x1, y1 = bbox 
            
            full_text = page.get_text("text", clip=fitz.Rect(x0, y0, x1, y1))
            for clipped_text, table_text in raw_text_list:
                if clipped_text in full_text:
                    full_text = full_text.replace(clipped_text, table_text)
                    
            chunk_temp.append(full_text)
            
    chunks = [Document(page_content=t) for t in chunk_temp]
    return chunks
