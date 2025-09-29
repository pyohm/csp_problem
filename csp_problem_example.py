from ortools.sat.python import cp_model

import random



# --- Data Generation ---

NUM_STUDENTS = 200 # Using 500 students as requested

NUM_CLASSROOMS = 6

CLASSROOM_IDS = list(range(NUM_CLASSROOMS))



# 1. Student Scores

student_scores = {f"S{i}": random.randint(50, 100) for i in range(NUM_STUDENTS)}



# 2. Student Dislikes (a few random pairs)

student_dislikes = set()

num_dislike_pairs = NUM_STUDENTS // 20 # E.g., 25 dislike pairs for 500 students

for _ in range(num_dislike_pairs):

    s1 = f"S{random.randint(0, NUM_STUDENTS - 1)}"

    s2 = f"S{random.randint(0, NUM_STUDENTS - 1)}"

    if s1 != s2 and (s2, s1) not in student_dislikes:

        student_dislikes.add((s1, s2))

print(f"Generated {len(student_dislikes)} dislike pairs.")



# 3. Student Subject Preferences and Teacher Subject Specializations

subjects = ["Math", "Science", "Literature", "History", "Art", "Music", "PE", "IT", "ForeignLanguage", "Economics"]



# Ensure we have enough unique subjects for classrooms

if NUM_CLASSROOMS > len(subjects):

    print("Warning: More classrooms than unique subjects. Some classrooms will have duplicate teacher subjects.")

    # If more classrooms than subjects, subjects will repeat

    classroom_teacher_subjects = {CLASSROOM_IDS[i]: subjects[i % len(subjects)] for i in range(NUM_CLASSROOMS)}

else:

    # Each classroom has a unique primary subject for its teacher

    classroom_teacher_subjects = {CLASSROOM_IDS[i]: subjects[i] for i in range(NUM_CLASSROOMS)}



student_preferences = {f"S{i}": random.choice(subjects) for i in range(NUM_STUDENTS)}



print(f"Classroom Teacher Subjects: {classroom_teacher_subjects}")



# --- CP-SAT Model Definition ---

model = cp_model.CpModel()



# --- Variables ---

# student_vars[student_id] will hold the assigned classroom ID for each student

student_vars = {}

for student_id in student_scores:

    student_vars[student_id] = model.NewIntVar(0, NUM_CLASSROOMS - 1, f"student_{student_id}_class")



# --- Constraints ---



# Constraint 1: Classroom Size Balance (Implicitly helps with average score)

# To ensure classes are not too unbalanced, we'll set min/max capacities.

# This also indirectly helps the average score constraint by ensuring each

# class has a similar number of students.

min_students_per_class = NUM_STUDENTS // NUM_CLASSROOMS - (NUM_STUDENTS % NUM_CLASSROOMS != 0) - 5

max_students_per_class = NUM_STUDENTS // NUM_CLASSROOMS + (NUM_STUDENTS % NUM_CLASSROOMS != 0) + 5

if min_students_per_class < 1: min_students_per_class = 1 # Ensure at least one student if possible



print(f"Min students per class: {min_students_per_class}, Max students per class: {max_students_per_class}")



# Create boolean variables for student-to-classroom assignments:

# assigned_to_class[student_id][class_id] is 1 if student_id is in class_id, 0 otherwise

assigned_to_class = {}

for student_id in student_scores:

    assigned_to_class[student_id] = {}

    for c_id in CLASSROOM_IDS:

        assigned_to_class[student_id][c_id] = model.NewBoolVar(f"student_{student_id}_in_class_{c_id}")

        # Link student_vars to these boolean variables:

        model.Add(student_vars[student_id] == c_id).OnlyEnforceIf(assigned_to_class[student_id][c_id])

        model.Add(student_vars[student_id] != c_id).OnlyEnforceIf(assigned_to_class[student_id][c_id].Not())





# Sum of students in each class

for c_id in CLASSROOM_IDS:

    students_in_this_class = [assigned_to_class[s_id][c_id] for s_id in student_scores]

    model.Add(sum(students_in_this_class) >= min_students_per_class)

    model.Add(sum(students_in_this_class) <= max_students_per_class)





# Constraint 2: Student Dislikes

for s1, s2 in student_dislikes:

    model.Add(student_vars[s1] != student_vars[s2])



# Constraint 3: Subject Preference Matching

# Students who like a subject should be in a classroom whose teacher teaches that subject.

for student_id, student_fav_subject in student_preferences.items():

    # If student_vars[student_id] == c_id, then classroom_teacher_subjects[c_id] must be student_fav_subject

    for c_id in CLASSROOM_IDS:

        if classroom_teacher_subjects[c_id] != student_fav_subject:

            # If the classroom's teacher does NOT teach the student's favorite subject,

            # then the student CANNOT be assigned to this classroom.

            model.Add(student_vars[student_id] != c_id)





# Constraint 4: Score Balancing (Optimization Objective)

# This is the most involved part. We want to minimize the variance or range of

# classroom average scores.

#

# Strategy:

# 1. Create a variable for the total score in each classroom.

# 2. Minimize the difference between the max and min total classroom scores.

#    (This is easier to model than minimizing variance directly in CP-SAT).



classroom_total_scores = {}

for c_id in CLASSROOM_IDS:

    # Calculate the sum of scores for students assigned to this classroom

    # The sum is `sum(assigned_to_class[s_id][c_id] * student_scores[s_id] for all students)`

    # where assigned_to_class[s_id][c_id] is 1 if student s_id is in class c_id.

    sum_expr = sum(assigned_to_class[s_id][c_id] * student_scores[s_id]

                   for s_id in student_scores)

    classroom_total_scores[c_id] = model.NewIntVar(0, sum(student_scores.values()), f"class_{c_id}_total_score")

    model.Add(classroom_total_scores[c_id] == sum_expr)





# Define variables for the minimum and maximum total scores across all classrooms

min_total_score = model.NewIntVar(0, sum(student_scores.values()), "min_total_score")

max_total_score = model.NewIntVar(0, sum(student_scores.values()), "max_total_score")



model.AddMinEquality(min_total_score, list(classroom_total_scores.values()))

model.AddMaxEquality(max_total_score, list(classroom_total_scores.values()))



# Objective: Minimize the range of total scores (max_total_score - min_total_score)

model.Minimize(max_total_score - min_total_score)





# --- Solve the Model ---

solver = cp_model.CpSolver()

# Set a time limit for finding a solution, especially for large problems

solver.parameters.max_time_in_seconds = 60.0 # 1 minute time limit

solver.parameters.num_workers = 8 # Use multiple cores for parallel search



print("\nSolving the model...")

status = solver.Solve(model)



# --- Display Results ---

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:

    print(f"\nSolution found (Status: {solver.StatusName(status)})")

    print(f"  - Objective value (Min/Max score difference): {solver.ObjectiveValue()}")

    print(f"  - Wall time: {solver.WallTime():.2f}s")

    print(f"  - Solve time: {solver.ProtoStats().solve_time_in_seconds:.2f}s")



    classroom_assignments = {c_id: [] for c_id in CLASSROOM_IDS}

    for student_id in student_scores:

        assigned_class = solver.Value(student_vars[student_id])

        classroom_assignments[assigned_class].append(student_id)



    total_overall_score = sum(student_scores.values())

    print(f"\nTotal Students: {NUM_STUDENTS}, Total Classrooms: {NUM_CLASSROOMS}")

    print(f"Overall Average Student Score: {total_overall_score / NUM_STUDENTS:.2f}")

    print(f"Ideal Students per Class: {NUM_STUDENTS / NUM_CLASSROOMS:.2f}")

    print(f"Ideal Total Score per Class: {total_overall_score / NUM_CLASSROOMS:.2f}\n")





    for c_id in CLASSROOM_IDS:

        students_in_class = classroom_assignments[c_id]

        class_scores = [student_scores[s] for s in students_in_class]

        num_students = len(students_in_class)

        class_total_score = sum(class_scores)

        class_avg_score = class_total_score / num_students if num_students > 0 else 0



        print(f"--- Classroom {c_id} (Teacher Subject: {classroom_teacher_subjects[c_id]}) ---")

        print(f"  Number of Students: {num_students}")

        print(f"  Total Score: {class_total_score}")

        print(f"  Average Score: {class_avg_score:.2f}")



        # Verify dislike constraint for this class

        dislike_violations = []

        for i in range(len(students_in_class)):

            for j in range(i + 1, len(students_in_class)):

                s1, s2 = students_in_class[i], students_in_class[j]

                if (s1, s2) in student_dislikes or (s2, s1) in student_dislikes:

                    dislike_violations.append(f"({s1}, {s2})")

        if dislike_violations:

            print(f"  !!! WARNING: Disliked pairs found in Classroom {c_id}: {', '.join(dislike_violations)}")



        # Verify subject preference for this class

        subject_violations = []

        for s in students_in_class:

            if classroom_teacher_subjects[c_id] != student_preferences[s]:

                subject_violations.append(f"{s} (likes {student_preferences[s]})")

        if subject_violations:

            print(f"  !!! WARNING: Subject preference mismatches in Classroom {c_id}: {', '.join(subject_violations)}")

        print("-" * 30)



else:

    print(f"\nNo solution found (Status: {solver.StatusName(status)})")

    print(f"  - Wall time: {solver.WallTime():.2f}s")


    print("Consider relaxing some constraints or increasing the time limit.")