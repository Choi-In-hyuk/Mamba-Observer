import re
import random
from collections import Counter

def debug_listops_data():
    """ListOps 데이터 생성과 레이블 계산을 확인하는 함수"""
    
    print("=" * 60)
    print("ListOps 데이터 디버깅")
    print("=" * 60)
    
    # 1. 몇 개 샘플 생성해서 확인
    def generate_and_evaluate_samples(num_samples=10):
        operations = ['MAX', 'MIN', 'MED', 'SM']
        
        def generate_expression(depth=0, max_depth=3):  # 더 간단하게
            if depth >= max_depth or random.random() < 0.4:
                return str(random.randint(0, 9))
            
            op = random.choice(operations)
            num_args = random.randint(2, 4)  # 더 적은 인수
            args = []
            
            for _ in range(num_args):
                if random.random() < 0.3:  # 중첩 확률 낮춤
                    args.append(generate_expression(depth + 1, max_depth))
                else:
                    args.append(str(random.randint(0, 9)))
            
            return f"[{op} {' '.join(args)}]"
        
        def evaluate_expression_detailed(expr):
            """상세한 디버깅이 포함된 평가 함수"""
            print(f"\n표현식: {expr}")
            
            try:
                # 토큰화
                tokens = re.findall(r'\[|\]|[A-Z]+|\d+', expr)
                print(f"토큰: {tokens}")
                
                def parse_tokens(tokens, pos):
                    if tokens[pos] == '[':
                        pos += 1
                        op = tokens[pos]
                        pos += 1
                        args = []
                        
                        print(f"  연산: {op}")
                        
                        while pos < len(tokens) and tokens[pos] != ']':
                            if tokens[pos] == '[':
                                arg, pos = parse_tokens(tokens, pos)
                                args.append(arg)
                                print(f"    중첩 결과: {arg}")
                            else:
                                arg = int(tokens[pos])
                                args.append(arg)
                                print(f"    숫자: {arg}")
                                pos += 1
                        
                        if pos < len(tokens):
                            pos += 1  # Skip ']'
                        
                        print(f"  {op}의 인수들: {args}")
                        
                        # 연산 수행
                        if op == 'MAX':
                            result = max(args)
                        elif op == 'MIN':
                            result = min(args)
                        elif op == 'MED':
                            sorted_args = sorted(args)
                            result = sorted_args[len(sorted_args) // 2]
                        elif op == 'SM':
                            result = sum(args) % 10
                        else:
                            result = 0
                        
                        print(f"  {op} 결과: {result}")
                        return result, pos
                    else:
                        return int(tokens[pos]), pos + 1
                
                result, _ = parse_tokens(tokens, 0)
                print(f"최종 결과: {result}")
                return result
            except Exception as e:
                print(f"오류 발생: {e}")
                print(f"토큰: {tokens}")
                return 0
        
        print("\n샘플 데이터 생성 및 평가:")
        for i in range(num_samples):
            print(f"\n--- 샘플 {i+1} ---")
            expr = generate_expression()
            label = evaluate_expression_detailed(expr)
            print(f"데이터: {label}\t{expr}")
    
    # 2. 기존 데이터 파일 확인 (있다면)
    def check_existing_data():
        import os
        train_path = "./data/listops/train.txt"
        
        if os.path.exists(train_path):
            print("\n기존 데이터 파일 확인:")
            with open(train_path, 'r') as f:
                lines = f.readlines()[:10]  # 처음 10개만
                
            for i, line in enumerate(lines):
                print(f"\n--- 기존 데이터 {i+1} ---")
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    label = parts[0]
                    expr = parts[1]
                    print(f"파일 레이블: {label}")
                    print(f"표현식: {expr}")
                    
                    # 다시 계산해서 비교
                    try:
                        tokens = re.findall(r'\[|\]|[A-Z]+|\d+', expr)
                        
                        def parse_tokens(tokens, pos):
                            if tokens[pos] == '[':
                                pos += 1
                                op = tokens[pos]
                                pos += 1
                                args = []
                                
                                while pos < len(tokens) and tokens[pos] != ']':
                                    if tokens[pos] == '[':
                                        arg, pos = parse_tokens(tokens, pos)
                                        args.append(arg)
                                    else:
                                        args.append(int(tokens[pos]))
                                        pos += 1
                                
                                if pos < len(tokens):
                                    pos += 1  # Skip ']'
                                
                                if op == 'MAX':
                                    result = max(args)
                                elif op == 'MIN':
                                    result = min(args)
                                elif op == 'MED':
                                    result = sorted(args)[len(args) // 2]
                                elif op == 'SM':
                                    result = sum(args) % 10
                                else:
                                    result = 0
                                
                                return result, pos
                            else:
                                return int(tokens[pos]), pos + 1
                        
                        calculated_label, _ = parse_tokens(tokens, 0)
                        print(f"계산된 레이블: {calculated_label}")
                        
                        if str(calculated_label) != label:
                            print(f"⚠️  레이블 불일치! 파일: {label}, 계산: {calculated_label}")
                        else:
                            print(f"✅ 레이블 일치")
                            
                    except Exception as e:
                        print(f"❌ 계산 오류: {e}")
        else:
            print("기존 데이터 파일이 없습니다.")
    
    # 3. 데이터셋 클래스 테스트
    def test_dataset_class():
        print("\n" + "="*40)
        print("데이터셋 클래스 테스트")
        print("="*40)
        
        # 임시 데이터 파일 생성
        temp_data = [
            "5\t[MAX 2 5 3]",
            "2\t[MIN 7 2 9]", 
            "4\t[MED 1 4 7 9]",
            "3\t[SM 1 2 3 7]",  # (1+2+3+7) % 10 = 3
            "8\t[MAX [MIN 3 8] 6]"
        ]
        
        with open("temp_listops.txt", "w") as f:
            for line in temp_data:
                f.write(line + "\n")
        
        # 데이터셋 로드 테스트
        try:
            from 기존_코드 import ListOpsDataset  # 실제로는 import 경로 수정 필요
            dataset = ListOpsDataset("temp_listops.txt", max_length=100)
            
            print(f"어휘 크기: {dataset.vocab_size}")
            print(f"데이터 개수: {len(dataset)}")
            print(f"어휘: {dataset.vocab[:20]}...")  # 처음 20개만
            
            # 몇 개 샘플 확인
            for i in range(min(5, len(dataset))):
                tokens, label = dataset[i]
                print(f"\n샘플 {i+1}:")
                print(f"  원본: {temp_data[i]}")
                print(f"  토큰: {tokens[:20]}...")  # 처음 20개만
                print(f"  레이블: {label}")
                
                # 토큰을 다시 문자로 변환해서 확인
                decoded = ''.join([dataset.idx_to_char[idx.item()] for idx in tokens if idx.item() < len(dataset.idx_to_char)])
                print(f"  디코드: {decoded}")
                
        except Exception as e:
            print(f"데이터셋 테스트 오류: {e}")
        
        # 임시 파일 삭제
        import os
        if os.path.exists("temp_listops.txt"):
            os.remove("temp_listops.txt")
    
    # 4. 레이블 분포 확인
    def check_label_distribution():
        print("\n" + "="*40) 
        print("레이블 분포 확인")
        print("="*40)
        
        # 1000개 샘플 생성해서 분포 확인
        labels = []
        for _ in range(1000):
            operations = ['MAX', 'MIN', 'MED', 'SM']
            
            def generate_simple():
                op = random.choice(operations)
                args = [random.randint(0, 9) for _ in range(random.randint(2, 5))]
                
                if op == 'MAX':
                    result = max(args)
                elif op == 'MIN':
                    result = min(args)
                elif op == 'MED':
                    result = sorted(args)[len(args) // 2]
                elif op == 'SM':
                    result = sum(args) % 10
                
                return result
            
            labels.append(generate_simple())
        
        distribution = Counter(labels)
        print("레이블 분포:")
        for label in range(10):
            count = distribution.get(label, 0)
            percentage = count / len(labels) * 100
            print(f"  {label}: {count}개 ({percentage:.1f}%)")
        
        # 너무 불균형하면 문제
        min_count = min(distribution.values())
        max_count = max(distribution.values())
        if max_count / min_count > 5:
            print("⚠️  레이블 분포가 매우 불균형합니다!")
        else:
            print("✅ 레이블 분포가 비교적 균형적입니다.")
    
    # 모든 테스트 실행
    generate_and_evaluate_samples(5)
    check_existing_data()
    test_dataset_class()
    check_label_distribution()

# 실행
if __name__ == "__main__":
    debug_listops_data()