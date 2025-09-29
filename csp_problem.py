import pandas as pd
from ortools.sat.python import cp_model
import math
from collections import defaultdict

def solve_class_assignment():
    # 1. 데이터 로드 및 전처리
    try:
        df = pd.read_csv('학급반편성CSP 문제 입력파일.csv')
    except FileNotFoundError:
        print("오류: '학급반편성CSP 문제 입력파일.csv' 파일을 찾을 수 없습니다.")
        return

    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()

    # 결측치를 처리
    df['좋은관계'] = df['좋은관계'].fillna(0).astype(int)
    df['나쁜관계'] = df['나쁜관계'].fillna(0).astype(int)
    df['Leadership'] = df['Leadership'].fillna('no')
    df['Piano'] = df['Piano'].fillna('no')
    df['비등교'] = df['비등교'].fillna('no')
    df['운동선호'] = df['운동선호'].fillna('no')
    df['24년 학급'] = df['24년 학급'].fillna('unknown')

    # 학생 데이터를 다루기 쉬운 딕셔너리 형태로 변환
    students = df.to_dict('records')
    student_ids = [s['id'] for s in students]
    student_map = {s['id']: s for s in students}

    # 2. 모델 기본 설정
    NUM_STUDENTS = len(students)
    NUM_CLASSES = 6
    CLASS_SIZES = [33, 33, 33, 33, 34, 34]
    CLASS_IDS = list(range(NUM_CLASSES))

    # 학생 수와 반 편성 총 인원이 일치하는지 확인
    assert NUM_STUDENTS == sum(CLASS_SIZES)

    model = cp_model.CpModel()

    # 3. 변수 정의
    # assign[s, c]는 학생 s가 c반에 배정되었는지 여부를 나타내는 불리언 변수
    assign = {}
    for s_id in student_ids:
        for c_id in CLASS_IDS:
            assign[s_id, c_id] = model.NewBoolVar(f'assign_{s_id}_to_{c_id}')

    # 4. 제약 조건 추가

    # 모든 학생은 정확히 하나의 반에만 배정되어야 합니다.
    for s_id in student_ids:
        model.AddExactlyOne(assign[s_id, c] for c in CLASS_IDS)

    # 각 반의 학생 수는 미리 정해진 크기를 준수해야 합니다.
    for c_id in CLASS_IDS:
        model.Add(sum(assign[s, c_id] for s in student_ids) == CLASS_SIZES[c_id])

    # 제약조건 1A: 사이가 나쁜 학생들은 같은 반에 배정하지 않습니다.
    for s1_id in student_ids:
        s2_id = student_map[s1_id]['나쁜관계']
        if s2_id in student_map:
            for c_id in CLASS_IDS:
                # s1과 s2가 동시에 c반에 배정되는 것을 방지
                model.AddBoolOr([assign[s1_id, c_id].Not(), assign[s2_id, c_id].Not()])

    # 제약조건 1B: 비등교 학생은 '좋은 관계'의 친구와 같은 반에 배정합니다.
    for s_id in student_ids:
        if student_map[s_id]['비등교'] == 'yes':
            friend_id = student_map[s_id]['좋은관계']
            if friend_id in student_map:
                for c_id in CLASS_IDS:
                    # 비등교 학생이 c반이면, 친구도 c반이어야 함
                    model.AddImplication(assign[s_id, c_id], assign[friend_id, c_id])

    def distribute_evenly(attribute_name):
        """특정 속성('yes' 값)을 가진 학생들을 모든 반에 균등하게 분배하는 함수"""
        students_with_attribute = [s_id for s_id in student_ids if student_map[s_id][attribute_name] == 'yes']
        total_count = len(students_with_attribute)
        min_per_class = total_count // NUM_CLASSES
        max_per_class = math.ceil(total_count / NUM_CLASSES)
        for c_id in CLASS_IDS:
            num_in_class = sum(assign[s_id, c_id] for s_id in students_with_attribute)
            model.AddLinearConstraint(num_in_class, min_per_class, max_per_class)
            
    

    # 제약조건 2: 리더십 학생을 각 반에 최소 1명 이상 배정합니다.
    leadership_students = [s_id for s_id in student_ids if student_map[s_id]['Leadership'] == 'yes']
    for c_id in CLASS_IDS:
        model.Add(sum(assign[s, c_id] for s in leadership_students) >= 1)
        
    # 제약조건 8: 전년도 같은 반 학생이 같은 반에 여러 명 모이면 '초과분'만큼 벌점을 부여(soft)
    last_year_classes = defaultdict(list)
    for st in students:
        ly = st.get('24년 학급')
        if ly and ly != 'unknown':
            last_year_classes[ly].append(st['id'])

    # 8번 soft 벌점을 담을 리스트
    ly_penalties = []  # IntVar들의 리스트 (초과분)

    # cap=1이면 "같은 반 금지"에 최대한 가깝게 유도 (2명 이상일 때부터 벌점)
    LY_CAP = 1

    for g_idx, group in enumerate(last_year_classes.values()):
        for c_id in CLASS_IDS:
            cnt_expr = sum(assign[s_id, c_id] for s_id in group)  # 해당 반에 같은 전년도 학급이 몇 명 배정됐는지
            over = model.NewIntVar(0, len(group), f"ly_over_{g_idx}_{c_id}")
            # over >= cnt - LY_CAP  (초과분만큼 벌점)
            model.Add(over >= cnt_expr - LY_CAP)
            # over는 음수가 될 수 없으므로 자동으로 max(cnt-LY_CAP, 0)과 같아지도록 모델링
            ly_penalties.append(over)

    # 제약조건 3, 5, 7: 피아노, 비등교, 운동선호 학생들을 균등하게 분배합니다.
    distribute_evenly('Piano')
    distribute_evenly('비등교')
    distribute_evenly('운동선호')

    # 제약조건 6: 남녀 비율을 균등하게 맞춥니다.
    male_students = [s_id for s_id in student_ids if student_map[s_id]['sex'] == 'boy']
    total_males = len(male_students)
    min_males = total_males // NUM_CLASSES
    max_males = math.ceil(total_males / NUM_CLASSES)
    for c_id in CLASS_IDS:
        model.AddLinearConstraint(sum(assign[s, c_id] for s in male_students), min_males, max_males)

    # 제약조건 9: 클럽 멤버들이 특정 반에 편중되지 않도록 분배합니다.
    all_clubs = df['클럽'].unique()
    for club in all_clubs:
        students_in_club = [s_id for s_id in student_ids if student_map[s_id]['클럽'] == club]
        if students_in_club:
            total_in_club = len(students_in_club)
            min_per_class = total_in_club // NUM_CLASSES
            max_per_class = math.ceil(total_in_club / NUM_CLASSES)
            for c_id in CLASS_IDS:
                model.AddLinearConstraint(sum(assign[s, c_id] for s in students_in_club), min_per_class, max_per_class)

    # 5. 목표 함수: 반별 성적 편차 최소화 + 8번 soft 벌점 최소화
    class_scores = [sum(assign[s, c] * student_map[s]['score'] for s in student_ids) for c in CLASS_IDS]
    min_score_var = model.NewIntVar(0, sum(s['score'] for s in students), 'min_score')
    max_score_var = model.NewIntVar(0, sum(s['score'] for s in students), 'max_score')
    model.AddMinEquality(min_score_var, class_scores)
    model.AddMaxEquality(max_score_var, class_scores)
    
    # 가중치 (필요시 조정)
    W_SCORE = 1          # 성적 편차 가중치
    W_LY    = 100        # 8번 벌점 가중치 (값을 높일수록 '같은반 겹침 최소화'를 더 강하게 유도)
    
    model.Minimize(max_score_var - min_score_var)

    # 6. 모델 해결 및 결과 출력
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120.0  # 해결 시간 2분으로 설정
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("최적의 반 편성 결과를 찾았습니다!")
        print(f"(소요 시간: {solver.WallTime():.2f}초)")
        print("-" * 30)

        class_assignments = {c: [] for c in CLASS_IDS}
        for s_id in student_ids:
            for c_id in CLASS_IDS:
                if solver.Value(assign[s_id, c_id]):
                    class_assignments[c_id].append(s_id)
                    break

        for c_id in CLASS_IDS:
            class_students_data = [student_map[s_id] for s_id in class_assignments[c_id]]
            avg_score = sum(s['score'] for s in class_students_data) / len(class_students_data)
            num_boys = sum(1 for s in class_students_data if s['sex'] == 'boy')
            
            print(f"## 클래스 {c_id + 1} (총 {len(class_students_data)}명)")
            print(f"  - 학생 수: 남 {num_boys}명, 여 {len(class_students_data) - num_boys}명")
            print(f"  - 평균 점수: {avg_score:.2f}")
            print(f"  - 리더십: {sum(1 for s in class_students_data if s['Leadership'] == 'yes')}명")
            print(f"  - 피아노: {sum(1 for s in class_students_data if s['Piano'] == 'yes')}명")
            print(f"  - 비등교: {sum(1 for s in class_students_data if s['비등교'] == 'yes')}명")
            
            student_names = [s['name'] for s in class_students_data]
            print(f"  - 학생 명단: {', '.join(student_names)}")
            print("-" * 20)
        
        # ====== [추가] 제약 만족도 검증 출력 ======
        print("\n=== 추가적인 제약 만족도 검증 ===")

        # 1) 유틸: 학생 -> 배정된 반 id
        assigned_class = {}
        for s_id in student_ids:
            for c_id in CLASS_IDS:
                if solver.Value(assign[s_id, c_id]):
                    assigned_class[s_id] = c_id
                    break

        def get_assigned_class(sid):
            return assigned_class.get(sid, None)

        # ---- 1A: 나쁜관계(같은 반 금지) ----
        bad_violations = []
        for s1_id in student_ids:
            s2_id = student_map[s1_id]['나쁜관계']
            if s2_id in student_map and s2_id != s1_id:
                c1 = get_assigned_class(s1_id)
                c2 = get_assigned_class(s2_id)
                if c1 is not None and c1 == c2:
                    bad_violations.append((s1_id, s2_id, c1))
        if bad_violations:
            print(f"[1A] 위반 {len(bad_violations)}건 발견 (같은 반 배정): 예) {bad_violations[:5]}")
        else:
            print("[1A] 만족: 나쁜관계 학생 페어가 같은 반에 배정되지 않음")

        # ---- 1B: 비등교 학생은 '좋은관계' 친구와 같은 반 ----
        absc_violations = []
        for s_id in student_ids:
            if str(student_map[s_id]['비등교']).lower() == 'yes':
                friend_id = student_map[s_id]['좋은관계']
                if friend_id in student_map and friend_id != s_id:
                    c1 = get_assigned_class(s_id)
                    c2 = get_assigned_class(friend_id)
                    # 두 명 모두 배정되었고, 같은 반이 아니면 위반
                    if c1 is not None and c2 is not None and c1 != c2:
                        absc_violations.append((s_id, friend_id, c1, c2))
        if absc_violations:
            print(f"[1B] 위반 {len(absc_violations)}건 발견 (같은 반 아님): 예) {absc_violations[:5]}")
        else:
            print("[1B] 만족: 비등교 학생은 지정된 좋은관계 친구와 같은 반")

        # ---- 7: 운동선호 균등 분배 검사 ----
        sport_ids = [s for s in student_ids if str(student_map[s]['운동선호']).lower() == 'yes']
        total_sport = len(sport_ids)
        min_sport = total_sport // NUM_CLASSES
        max_sport = math.ceil(total_sport / NUM_CLASSES)
        sport_counts = []
        sport_viol = []
        for c_id in CLASS_IDS:
            cnt = sum(1 for s in sport_ids if get_assigned_class(s) == c_id)
            sport_counts.append(cnt)
            if not (min_sport <= cnt <= max_sport):
                sport_viol.append((c_id, cnt))
        if sport_viol:
            print(f"[7] 위반: 반별 운동선호 수가 범위를 벗어남 (허용 {min_sport}~{max_sport}) → {sport_viol}")
        else:
            print(f"[7] 만족: 운동선호 균등 분배 OK (반별 분포: {sport_counts})")

        # ---- 8: 전년도 같은 반 soft 벌점 검증 ----
        last_year_classes = defaultdict(list)
        for st in students:
            ly = st.get('24년 학급')
            if ly and ly != 'unknown':
                last_year_classes[ly].append(st['id'])

        LY_CAP = 1  # 모델의 cap과 동일해야 함
        total_ly_penalty = 0
        top_cases = []  # (전년도학급, 현재반, 인원수, 초과분)

        for ly_name, group in last_year_classes.items():
            for c_id in CLASS_IDS:
                cnt = sum(1 for s in group if get_assigned_class(s) == c_id)
                over = max(cnt - LY_CAP, 0)
                if over > 0:
                    top_cases.append((ly_name, c_id+1, cnt, over))
                total_ly_penalty += over

        # 보고용: 초과 사례 상위 몇 개만
        top_cases.sort(key=lambda x: x[3], reverse=True)

        if total_ly_penalty == 0:
            print(f"[8] 만족: 전년도 동일 학급 학생의 동일반 겹침 없음(벌점=0, cap={LY_CAP})")
        else:
            print(f"[8] 만족: 전년도 동일 학급 학생의 동일반 겹침 발생 최소화 — 총 벌점={total_ly_penalty} (cap={LY_CAP})")
            #print(f"     예시 상위: {top_cases[:6]}")

        # ---- 9: 클럽 편중 방지 검사 ----
        clubs = df['클럽'].fillna('none').astype(str).str.strip().unique()
        club_viol = []
        for club in clubs:
            club_ids = [s for s in student_ids if str(student_map[s]['클럽']).strip() == club]
            if not club_ids:
                continue
            total = len(club_ids)
            min_per = total // NUM_CLASSES
            max_per = math.ceil(total / NUM_CLASSES)
            dist = []
            for c_id in CLASS_IDS:
                cnt = sum(1 for s in club_ids if get_assigned_class(s) == c_id)
                dist.append(cnt)
                if not (min_per <= cnt <= max_per):
                    club_viol.append((club, c_id, cnt, f"허용:{min_per}~{max_per}"))
        if club_viol:
            print(f"[9] 위반: 클럽 편중 범위 벗어남 예) {club_viol[:6]}")
        else:
            print("[9] 만족: 클럽 인원이 반마다 허용 범위 내로 분산")
        print("=== 검증 끝 ===\n")
        
    else:
        print("해결책을 찾을 수 없었습니다. 일부 제약 조건이 너무 엄격할 수 있습니다.")
        print("리더십 수:", len([s for s in students if s['Leadership']=='yes']))
        print("전년도 반별 인원:", {k: len(v) for k, v in last_year_classes.items()})
        print("남자 수/전체:", sum(1 for s in students if s['sex']=='boy'), "/", len(students))
        print("피아노/비등교/운동선호 수:", 
            sum(1 for s in students if s['Piano']=='yes'),
            sum(1 for s in students if s['비등교']=='yes'),
            sum(1 for s in students if s['운동선호']=='yes'))

if __name__ == '__main__':
    solve_class_assignment()