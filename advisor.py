import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import time
from curriculum import create_curriculum  # IMPORT ADDED HERE

class AcademicAdvisor:
    def __init__(self, curriculum_graph):
        self.curriculum = curriculum_graph
        self.courses = list(curriculum_graph.nodes())
        self.course_features = self._extract_course_features()
        self.grade_predictor = self._train_grade_predictor()
    
    def _extract_course_features(self):
        features = {}
        for course in self.courses:
            data = self.curriculum.nodes[course]
            features[course] = {
                "level": data["level"],
                "credits": data["credits"],
                "interest_ai": 1 if data["interest"] == "AI" else 0,
                "interest_security": 1 if data["interest"] == "Security" else 0,
                "interest_datascience": 1 if data["interest"] == "Data Science" else 0,
                "interest_core": 1 if data["interest"] == "Core" else 0,
                "interest_math": 1 if data["interest"] == "Math" else 0,
                "interest_systems": 1 if data["interest"] == "Systems" else 0,
                "prereq_count": len(list(self.curriculum.predecessors(course)))
            }
        return features
    
    def _train_grade_predictor(self, n_samples=5000):
        X, y = [], []
        
        for _ in range(n_samples):
            gpa = np.clip(np.random.normal(3.0, 0.5), 2.0, 4.0)
            term = np.random.randint(1, 8)
            interests = np.random.choice(["AI", "Security", "Data Science", "Core", "Systems"], 2, replace=False)
            
            for course, feats in self.course_features.items():
                student_feats = [
                    gpa,
                    term,
                    1 if "AI" in interests else 0,
                    1 if "Security" in interests else 0,
                    1 if "Data Science" in interests else 0,
                    1 if "Core" in interests else 0,
                    1 if "Systems" in interests else 0,
                ]
                course_feats = list(feats.values())
                combined = student_feats + course_feats
                
                # Realistic grade simulation with interest boost
                base_grade = gpa * (1 - 0.03 * feats["level"])
                if self.curriculum.nodes[course]["interest"] in interests:
                    base_grade += 0.6  # Significant boost for interest match
                
                noise = np.random.normal(0, 0.3)
                grade = np.clip(base_grade + noise, 0, 4.0)
                
                X.append(combined)
                y.append(grade)
        
        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        print(f"Grade predictor trained with R²={score:.3f}")
        return model
    
    def predict_grade(self, student, course):
        student_feats = [
            student["gpa"],
            student["term"],
            1 if "AI" in student["interests"] else 0,
            1 if "Security" in student["interests"] else 0,
            1 if "Data Science" in student["interests"] else 0,
            1 if "Core" in student["interests"] else 0,
            1 if "Systems" in student["interests"] else 0,
        ]
        
        course_feats = list(self.course_features[course].values())
        combined = np.array([student_feats + course_feats])
        
        return self.grade_predictor.predict(combined)[0]
    
    def get_eligible_courses(self, student):
        completed = list(student["completed"].keys()) if isinstance(student["completed"], dict) else student["completed"]
        eligible = []
        
        for course in self.courses:
            if course in completed:
                continue
                
            prereqs = list(self.curriculum.predecessors(course))
            if all(p in completed for p in prereqs):
                eligible.append(course)
        
        return eligible
    
    def heuristic_recommendation(self, student, max_courses=4):
        eligible = self.get_eligible_courses(student)
        if not eligible:
            return []
        
        scores = []
        for course in eligible:
            course_interest = self.curriculum.nodes[course]["interest"]
            
            # Interest alignment (primary factor)
            interest_score = 3.0 if course_interest in student["interests"] else 0.1
            
            # Predicted grade
            grade_score = self.predict_grade(student, course)
            
            # Course level alignment
            level_diff = abs(self.curriculum.nodes[course]["level"] - student["term"])
            level_score = 1.0 / (1 + level_diff)
            
            # Progress toward graduation
            progress_score = self.curriculum.nodes[course]["level"] / 4.0
            
            # Major alignment bonus
            major_bonus = 0
            if student["major"] == "Data Science" and course_interest == "Data Science":
                major_bonus = 1.0
            elif student["major"] == "AI Engineering" and course_interest == "AI":
                major_bonus = 1.0
            elif student["major"] == "Cyber Security" and course_interest == "Security":
                major_bonus = 1.0
            
            total_score = (
                interest_score * 0.6 + 
                grade_score * 0.2 + 
                level_score * 0.1 + 
                progress_score * 0.1 +
                major_bonus
            )
            
            scores.append((course, total_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [course for course, _ in scores[:max_courses]]
    
    def rl_recommendation(self, student, max_courses=4, num_simulations=2000):
        eligible = self.get_eligible_courses(student)
        if not eligible:
            return []
        
        course_scores = {course: 0 for course in eligible}
        course_counts = {course: 0 for course in eligible}
        
        for _ in range(num_simulations):
            # Select courses with probability based on interest
            weights = []
            for course in eligible:
                if self.curriculum.nodes[course]["interest"] in student["interests"]:
                    weights.append(5.0)  # Higher weight for interests
                else:
                    weights.append(1.0)
            
            weights = np.array(weights) / sum(weights)
            selected = np.random.choice(
                eligible, 
                size=min(max_courses, len(eligible)), 
                replace=False,
                p=weights
            )
            
            # Simulate outcomes
            reward = 0
            for course in selected:
                grade = self.predict_grade(student, course)
                
                # Reward components
                interest_reward = 2.0 if self.curriculum.nodes[course]["interest"] in student["interests"] else -0.5
                grade_reward = grade / 2.0  # Scaled reward
                progress_reward = self.curriculum.nodes[course]["level"] / 5.0
                
                # Major alignment bonus
                major_bonus = 0
                if student["major"] == "Data Science" and self.curriculum.nodes[course]["interest"] == "Data Science":
                    major_bonus = 1.5
                elif student["major"] == "AI Engineering" and self.curriculum.nodes[course]["interest"] == "AI":
                    major_bonus = 1.5
                elif student["major"] == "Cyber Security" and self.curriculum.nodes[course]["interest"] == "Security":
                    major_bonus = 1.5
                
                reward += interest_reward + grade_reward + progress_reward + major_bonus
            
            # Update scores
            for course in selected:
                course_scores[course] += reward
                course_counts[course] += 1
        
        # Calculate average scores
        avg_scores = []
        for course in eligible:
            if course_counts[course] > 0:
                avg_score = course_scores[course] / course_counts[course]
                avg_scores.append((course, avg_score))
        
        avg_scores.sort(key=lambda x: x[1], reverse=True)
        return [course for course, _ in avg_scores[:max_courses]]
    
    def evaluate_recommendations(self, students, method="heuristic", max_courses=4):
        results = []
        start_time = time.time()
        
        for _, student in tqdm(students.iterrows(), total=len(students), desc=f"Evaluating {method}"):
            if method == "heuristic":
                recommended = self.heuristic_recommendation(student, max_courses)
            else:
                recommended = self.rl_recommendation(student, max_courses)
            
            if not recommended:
                results.append({
                    "student_id": student["student_id"],
                    "gpa": student["gpa"],
                    "interests": student["interests"],
                    "major": student["major"],
                    "interest_alignment": 0,
                    "predicted_gpa": 0,
                    "progress_score": 0,
                    "recommended_courses": ""
                })
                continue
            
            # Calculate interest alignment
            interest_matches = 0
            for course in recommended:
                if self.curriculum.nodes[course]["interest"] in student["interests"]:
                    interest_matches += 1
            interest_alignment = interest_matches / len(recommended)
            
            # Calculate predicted GPA
            predicted_grades = [self.predict_grade(student, course) for course in recommended]
            predicted_gpa = np.mean(predicted_grades) if predicted_grades else 0
            
            # Calculate progress score
            progress_score = np.mean([self.curriculum.nodes[course]["level"] / 4.0 for course in recommended])
            
            results.append({
                "student_id": student["student_id"],
                "gpa": student["gpa"],
                "interests": student["interests"],
                "major": student["major"],
                "interest_alignment": interest_alignment,
                "predicted_gpa": predicted_gpa,
                "progress_score": progress_score,
                "recommended_courses": ", ".join(recommended)
            })
        
        elapsed = time.time() - start_time
        print(f"{method} evaluation completed in {elapsed:.2f} seconds")
        return pd.DataFrame(results)
    
    def visualize_recommendations(self, results, method="heuristic", filename=None):
        if not filename:
            filename = f"{method}_metrics.png"
        
        plt.figure(figsize=(14, 10))
        
        # Interest Alignment
        plt.subplot(2, 2, 1)
        plt.hist(results["interest_alignment"], bins=np.arange(0, 1.1, 0.1), 
                color="green", alpha=0.7, edgecolor='black')
        plt.title("Interest Alignment Distribution")
        plt.xlabel("Alignment Score")
        plt.ylabel("Number of Students")
        plt.grid(True, alpha=0.3)
        
        # Predicted GPA vs Current GPA
        plt.subplot(2, 2, 2)
        plt.scatter(results["gpa"], results["predicted_gpa"], alpha=0.6, color="blue")
        plt.plot([2, 4], [2, 4], 'r--')
        plt.title("Predicted GPA vs Current GPA")
        plt.xlabel("Current GPA")
        plt.ylabel("Predicted GPA")
        plt.grid(True, alpha=0.3)
        
        # Progress Score
        plt.subplot(2, 2, 3)
        plt.boxplot(results["progress_score"], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
        plt.title("Progress Toward Graduation")
        plt.ylabel("Progress Score")
        plt.grid(True, alpha=0.3, axis="y")
        
        # Success Rate
        plt.subplot(2, 2, 4)
        success_rate = ((results["interest_alignment"] > 0.5) & 
                       (results["predicted_gpa"] > 2.7)).mean()
        failure_rate = 1 - success_rate
        plt.bar(["Success", "Failure"], [success_rate, failure_rate], 
               color=["green", "red"])
        plt.title(f"Recommendation Success Rate: {success_rate:.1%}")
        plt.ylabel("Percentage")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis="y")
        
        plt.suptitle(f"{method} Recommendation Metrics", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Visualization saved to {filename}")

def load_student_data(filename="students.json"):
    with open(filename, 'r') as f:
        students = json.load(f)
    return pd.DataFrame(students)

def main():
    # Load curriculum and students
    curriculum = create_curriculum()
    students_df = load_student_data()
    
    # Initialize advisor
    advisor = AcademicAdvisor(curriculum)
    
    # Sample recommendations
    sample_students = students_df.sample(5, random_state=42)
    print("\nSample Recommendations:")
    for _, student in sample_students.iterrows():
        print(f"\nStudent {student['student_id']} (GPA: {student['gpa']}, "
              f"Interests: {student['interests']}, Major: {student['major']})")
        print(f"Completed: {list(student['completed'].keys())}")
        
        # Heuristic recommendation
        heuristic_rec = advisor.heuristic_recommendation(student)
        print(f"Heuristic Recommendation: {heuristic_rec}")
        
        # RL recommendation
        rl_rec = advisor.rl_recommendation(student)
        print(f"RL Recommendation: {rl_rec}")
    
    # Evaluate both methods
    heuristic_results = advisor.evaluate_recommendations(students_df, "heuristic")
    rl_results = advisor.evaluate_recommendations(students_df, "rl")
    
    # Save results
    heuristic_results.to_csv("heuristic_recommendations.csv", index=False)
    rl_results.to_csv("rl_recommendations.csv", index=False)
    
    # Visualize results
    advisor.visualize_recommendations(heuristic_results, "Heuristic")
    advisor.visualize_recommendations(rl_results, "RL")
    
    # Compare methods
    print("\nPerformance Comparison:")
    print(f"Heuristic - Avg Interest Alignment: {heuristic_results['interest_alignment'].mean():.3f}")
    print(f"RL - Avg Interest Alignment: {rl_results['interest_alignment'].mean():.3f}")
    print(f"Heuristic - Avg Predicted GPA: {heuristic_results['predicted_gpa'].mean():.3f}")
    print(f"RL - Avg Predicted GPA: {rl_results['predicted_gpa'].mean():.3f}")
    print(f"Heuristic - Avg Progress: {heuristic_results['progress_score'].mean():.3f}")
    print(f"RL - Avg Progress: {rl_results['progress_score'].mean():.3f}")
    
    return advisor, heuristic_results, rl_results

if __name__ == "__main__":
    advisor, heuristic_results, rl_results = main()