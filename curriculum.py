import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

def create_curriculum():
    G = nx.DiGraph()
    
    # Enhanced course list with better interest tagging
    courses = [
        {"id": "CS101", "name": "Intro to Programming", "credits": 4, "interest": "Core", "level": 1},
        {"id": "CS201", "name": "Data Structures", "credits": 4, "interest": "Core", "level": 2},
        {"id": "CS301", "name": "Algorithms", "credits": 4, "interest": "Core", "level": 3},
        {"id": "MATH101", "name": "Calculus I", "credits": 3, "interest": "Math", "level": 1},
        {"id": "MATH201", "name": "Linear Algebra", "credits": 3, "interest": "Math", "level": 2},
        {"id": "AI401", "name": "Machine Learning", "credits": 4, "interest": "AI", "level": 4},
        {"id": "AI402", "name": "Deep Learning", "credits": 4, "interest": "AI", "level": 4},
        {"id": "SEC401", "name": "Network Security", "credits": 4, "interest": "Security", "level": 4},
        {"id": "DS401", "name": "Big Data Analytics", "credits": 4, "interest": "Data Science", "level": 4},
        {"id": "DS201", "name": "Data Visualization", "credits": 3, "interest": "Data Science", "level": 2},
        {"id": "CS401", "name": "Software Engineering", "credits": 4, "interest": "Core", "level": 4},
        {"id": "AI301", "name": "Intro to AI", "credits": 3, "interest": "AI", "level": 3},
        {"id": "SEC301", "name": "Cyber Security", "credits": 3, "interest": "Security", "level": 3},
        {"id": "ELEC1", "name": "Cloud Computing", "credits": 3, "interest": "Systems", "level": 3},
        {"id": "ELEC2", "name": "Blockchain", "credits": 3, "interest": "Security", "level": 3},
    ]
    
    for course in courses:
        G.add_node(course["id"], **course)
    
    prerequisites = [
        ("CS101", "CS201"),
        ("CS201", "CS301"),
        ("CS301", "AI401"),
        ("CS301", "AI402"),
        ("CS301", "SEC401"),
        ("MATH101", "MATH201"),
        ("MATH201", "AI401"),
        ("MATH201", "DS401"),
        ("CS201", "DS401"),
        ("CS201", "DS201"),
        ("CS301", "CS401"),
        ("CS201", "ELEC1"),
        ("CS201", "ELEC2"),
        ("CS201", "AI301"),
        ("CS201", "SEC301"),
    ]
    
    for src, dst in prerequisites:
        G.add_edge(src, dst)
    
    return G

def visualize_curriculum(G, filename="curriculum.png"):
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42)
    
    interest_colors = {
        "Core": "#1f77b4",
        "AI": "#2ca02c",
        "Security": "#d62728",
        "Data Science": "#9467bd",
        "Math": "#ff7f0e",
        "Systems": "#8c564b"
    }
    
    node_colors = [interest_colors.get(G.nodes[n]["interest"], "#17becf") for n in G.nodes]
    
    nx.draw_networkx(
        G, pos, 
        with_labels=True, 
        node_size=2000, 
        node_color=node_colors, 
        font_size=9,
        edge_color="gray",
        font_weight="bold"
    )
    
    # Create legend
    legend_handles = []
    for interest, color in interest_colors.items():
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color, markersize=10, label=interest))
    
    plt.legend(handles=legend_handles, loc='best', frameon=True)
    plt.title("University Curriculum Prerequisite Graph", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Curriculum visualization saved to {filename}")
    return filename

def generate_students(num_students=100):
    np.random.seed(42)
    students = []
    interests = ["AI", "Security", "Data Science", "Core", "Systems"]
    majors = ["Computer Science", "Data Science", "AI Engineering", "Cyber Security"]
    
    curriculum = create_curriculum()
    all_courses = list(curriculum.nodes)
    
    for i in range(num_students):
        student_id = f"S{i:03d}"
        base_gpa = np.clip(np.random.normal(3.0, 0.5), 2.0, 4.0)
        term = np.random.randint(1, 6)
        student_interests = list(np.random.choice(interests, size=2, replace=False))
        major = np.random.choice(majors)
        
        # Bias interests based on major
        if major == "Data Science":
            student_interests = ["Data Science"] + [np.random.choice([i for i in interests if i != "Data Science"])]
        elif major == "AI Engineering":
            student_interests = ["AI"] + [np.random.choice([i for i in interests if i != "AI"])]
        elif major == "Cyber Security":
            student_interests = ["Security"] + [np.random.choice([i for i in interests if i != "Security"])]
        
        completed = {}
        completed_courses = []
        
        max_completed = min(len(all_courses), int(term * 4 * 0.8))
        num_completed = np.random.randint(max(0, max_completed-3), max_completed+1)
        
        for course in nx.topological_sort(curriculum):
            if course in completed_courses:
                continue
                
            prereqs = list(curriculum.predecessors(course))
            if all(p in completed_courses for p in prereqs):
                # Higher probability for courses matching interests/major
                interest_match = 1.0 if curriculum.nodes[course]["interest"] in student_interests else 0.4
                if np.random.rand() < interest_match and len(completed_courses) < num_completed:
                    # Grade based on GPA + interest match + difficulty
                    interest_boost = 0.7 if curriculum.nodes[course]["interest"] in student_interests else 0.0
                    difficulty_penalty = curriculum.nodes[course]["level"] * 0.1
                    grade = np.clip(
                        np.random.normal(base_gpa + interest_boost - difficulty_penalty, 0.5),
                        0, 4.0
                    )
                    completed[course] = round(grade, 2)
                    completed_courses.append(course)
        
        # Calculate actual GPA
        if completed:
            actual_gpa = round(np.mean(list(completed.values())), 2)
        else:
            actual_gpa = 0.0
        
        students.append({
            "student_id": student_id,
            "gpa": actual_gpa,
            "term": term,
            "interests": student_interests,
            "major": major,
            "completed": completed
        })
    
    return pd.DataFrame(students)

def save_curriculum(G, filename="curriculum.csv"):
    courses = []
    for node, data in G.nodes(data=True):
        prereqs = list(G.predecessors(node))
        courses.append({
            "course_id": node,
            "course_name": data["name"],
            "credits": data["credits"],
            "interest": data["interest"],
            "level": data["level"],
            "prerequisites": ", ".join(prereqs) if prereqs else "None"
        })
    
    df = pd.DataFrame(courses)
    df.to_csv(filename, index=False)
    print(f"Curriculum saved to {filename}")
    return df

def save_student_data(students_df, filename="students.json"):
    # Convert to JSON for better nested structure handling
    students_dict = students_df.to_dict(orient='records')
    with open(filename, 'w') as f:
        json.dump(students_dict, f, indent=2)
    print(f"Student data saved to {filename}")

def main():
    G = create_curriculum()
    visualize_curriculum(G)
    save_curriculum(G)
    
    students_df = generate_students(100)
    save_student_data(students_df)
    
    # Print sample data
    print("\nSample curriculum data:")
    print(save_curriculum(G).head())
    
    print("\nSample student data:")
    print(students_df[['student_id', 'gpa', 'term', 'interests', 'major']].head())
    print("Completed courses sample:", students_df.iloc[0]['completed'])
    
    return G, students_df

if __name__ == "__main__":
    curriculum_graph, students_df = main()