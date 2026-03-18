import pandas as pd
import ast
import os

def preprocess_csv(file_path):
    try:
        # 인코딩 대응
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp949')

        if 'demographic_identifier' in df.columns and 'pronoun' in df.columns:
            def determine_pronoun(val):
                if pd.isna(val): return None
                try:
                    # '["male", "poor"]' 형태의 문자열을 실제 리스트로 변환
                    traits = ast.literal_eval(val) if isinstance(val, str) else val
                    
                    # 리스트 내에 'female'이 있으면 she, 'male'이 있으면 he (female 우선 확인)
                    if 'female' in traits:
                        return 'she'
                    elif 'male' in traits:
                        return 'he'
                except (ValueError, SyntaxError):
                    # 리스트 형태가 아닐 경우 기존 방식대로 텍스트 검색
                    if 'female' in str(val).lower(): return 'she'
                    if 'male' in str(val).lower(): return 'he'
                return None

            # pronoun 열 업데이트
            df['pronoun'] = df['demographic_identifier'].apply(determine_pronoun)
            
            # 저장
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"✅ 리스트 기반 처리 완료: {file_path}")
        else:
            print(f"⚠️ 필수 열이 없습니다: {file_path}")
            
    except Exception as e:
        print(f"❌ 에러 발생: {e}")

# 실행
data_path = 'data/lbox.csv'
preprocess_csv(data_path)