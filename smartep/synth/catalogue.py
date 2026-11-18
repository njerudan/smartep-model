"""Courses, reskilling catalogue and skill tags used by generator v2.

Courses are typical Kenyan higher-education technology courses. Each topic has a difficulty in [-1, 1]
(later weeks are harder). ``outdated`` terms are deprecated practice that signals low currency of
materials; ``current`` terms are their up-to-date counterparts.
"""

from __future__ import annotations

COURSES: dict[str, dict] = {
    "Python Programming": {
        "discipline": "Computing",
        "skills": ["python", "programming"],
        "topics": ["Variables and types", "Control flow", "Functions", "Lists and dictionaries", "Strings and files",
                   "Modules and packages", "Error handling", "Object-oriented programming", "Testing with pytest",
                   "Working with APIs", "Data analysis with pandas", "Project and code review"],
        "outdated": ["Python 2.7 print statements", "raw_input()", "string formatting with the % operator only"],
        "current": ["Python 3.12", "type hints", "virtual environments with venv", "f-strings"],
    },
    "Databases and SQL": {
        "discipline": "Information Technology",
        "skills": ["databases", "sql"],
        "topics": ["Relational model", "Keys and constraints", "Basic SELECT queries", "Joins", "Aggregation",
                   "Normalisation", "Indexes", "Transactions and ACID", "Stored procedures", "Query optimisation",
                   "NoSQL overview", "Database security"],
        "outdated": ["MySQL 5.0", "Microsoft Access as a production database", "MD5 password hashing"],
        "current": ["PostgreSQL 16", "parameterised queries", "bcrypt password hashing"],
    },
    "Cloud Computing and Docker": {
        "discipline": "Computing",
        "skills": ["cloud", "docker", "devops"],
        "topics": ["Cloud service models", "Virtual machines", "Linux basics", "Networking in the cloud",
                   "Containers with Docker", "Docker Compose", "Container registries", "CI/CD pipelines",
                   "Infrastructure as code", "Kubernetes basics", "Monitoring and logging", "Cost management"],
        "outdated": ["FTP deployment to a single server", "Vagrant-only workflows", "manual server configuration"],
        "current": ["Docker containers", "GitHub Actions", "Kubernetes", "Terraform"],
    },
    "Computer Networks": {
        "discipline": "Engineering",
        "skills": ["networking"],
        "topics": ["Network models", "Physical layer", "Ethernet and switching", "IP addressing", "Subnetting",
                   "Routing", "TCP and UDP", "DNS and DHCP", "Wireless networks", "Network security",
                   "IPv6", "Software-defined networking"],
        "outdated": ["WEP Wi-Fi encryption", "Windows XP networking", "hubs as the standard LAN device"],
        "current": ["WPA3", "IPv6", "TLS 1.3", "software-defined networking"],
    },
    "Web Development": {
        "discipline": "Computing",
        "skills": ["web development", "javascript"],
        "topics": ["HTML structure", "CSS layout", "Responsive design", "JavaScript basics", "DOM manipulation",
                   "Fetch and APIs", "Frontend frameworks", "Node.js servers", "Authentication", "Databases for the web",
                   "Deployment", "Web accessibility"],
        "outdated": ["Adobe Flash", "table-based page layout", "Internet Explorer 6 support"],
        "current": ["CSS grid", "ES modules", "OAuth 2.0", "WCAG 2.2"],
    },
    "Cybersecurity Fundamentals": {
        "discipline": "Information Technology",
        "skills": ["security"],
        "topics": ["Threat landscape", "Cryptography basics", "Authentication", "Access control", "Network attacks",
                   "Web application security", "Malware", "Incident response", "Security policies", "Risk assessment",
                   "Kenya Data Protection Act", "Penetration testing basics"],
        "outdated": ["SSL 3.0", "DES encryption", "password-only authentication"],
        "current": ["TLS 1.3", "AES-256", "multi-factor authentication", "OWASP Top 10 (2021)"],
    },
    "Data Structures and Algorithms": {
        "discipline": "Computing",
        "skills": ["algorithms", "programming"],
        "topics": ["Complexity analysis", "Arrays", "Linked lists", "Stacks and queues", "Recursion", "Sorting",
                   "Searching", "Hash tables", "Trees", "Heaps", "Graphs", "Dynamic programming"],
        "outdated": ["Turbo C++ 3.0", "conio.h examples"],
        "current": ["Python 3.12", "modern C++ (C++20)", "unit-tested implementations"],
    },
    "Introduction to Machine Learning": {
        "discipline": "Computing",
        "skills": ["machine learning", "ai", "python"],
        "topics": ["What is machine learning", "Data preparation", "Linear regression", "Classification",
                   "Model evaluation", "Decision trees", "Ensembles", "Clustering", "Neural networks",
                   "Responsible AI", "Large language models", "ML project"],
        "outdated": ["Theano", "MATLAB-only examples", "Weka GUI as the main tool"],
        "current": ["scikit-learn", "PyTorch", "responsible AI checklists", "large language models"],
    },
}

# Skills that address each gap area; used for catalogue tags and, in Phase 2, for matching.
AREA_SKILLS: dict[str, list[str]] = {
    "content_coverage": ["curriculum planning", "course design"],
    "content_currency": ["industry practice", "course design"],
    "resources": ["learning materials", "digital tools"],
    "video_delivery": ["video production", "digital tools"],
    "session_delivery": ["time management", "curriculum planning"],
    "assessment_feedback": ["assessment", "feedback"],
    "student_engagement": ["active learning", "student engagement"],
    "pedagogy": ["pedagogy", "active learning"],
    "technical_skill": ["subject expertise"],
}


def _entry(ref, source, title, provider, description, skills, level, mode, cost, duration, url):
    return {"catalogueRef": ref, "source": source, "title": title, "provider": provider, "description": description,
            "skills": skills, "level": level, "mode": mode, "durationText": duration, "cost": cost, "url": url}


RESKILLING_CATALOGUE: list[dict] = [
    _entry("cat_tpack", "classroom", "Teaching with Digital Tools (TPACK)", "The Classroom",
           "Plan technology-rich lessons that combine content, pedagogy and tools.",
           ["pedagogy", "digital tools", "learning materials"], "beginner", "online", "free", "4 weeks",
           "https://app.example.test/courses/tpack"),
    _entry("cat_aiml", "classroom", "AI and ML for Educators", "The Classroom",
           "Core machine-learning ideas and classroom uses of AI.",
           ["ai", "machine learning", "subject expertise"], "beginner", "online", "free", "5 weeks",
           "https://app.example.test/courses/ai-ml-educators"),
    _entry("cat_python", "classroom", "Python Programming", "The Classroom",
           "Modern Python 3 from basics to testing.", ["python", "programming", "subject expertise"],
           "intermediate", "online", "free", "8 weeks", "https://app.example.test/courses/python"),
    _entry("cat_sql", "classroom", "Databases and SQL", "The Classroom",
           "Relational design and SQL with PostgreSQL.", ["databases", "sql", "subject expertise"],
           "intermediate", "online", "free", "6 weeks", "https://app.example.test/courses/sql"),
    _entry("cat_cloud", "classroom", "Cloud and Docker", "The Classroom",
           "Containers, CI/CD and cloud basics.", ["cloud", "docker", "devops", "industry practice"],
           "intermediate", "online", "free", "6 weeks", "https://app.example.test/courses/cloud-docker"),
    _entry("cat_assess", "institution", "Assessment and Feedback Practice", "Centre for Teaching Excellence",
           "Design assessments and give timely, useful feedback.", ["assessment", "feedback"],
           "beginner", "hybrid", {"amount": 3000, "currency": "KES"}, "3 days",
           "https://institution.example.test/cte/assessment"),
    _entry("cat_video", "institution", "Video Production for Teaching", "School of Computing Short Courses",
           "Record, edit and caption short teaching videos.", ["video production", "digital tools"],
           "beginner", "in_person", {"amount": 5000, "currency": "KES"}, "2 days",
           "https://institution.example.test/short-courses/video"),
    _entry("cat_active", "institution", "Active Learning Strategies", "Centre for Teaching Excellence",
           "Techniques that get students working in class.", ["active learning", "student engagement", "pedagogy"],
           "beginner", "hybrid", "free", "4 sessions", "https://institution.example.test/cte/active-learning"),
    _entry("cat_plan", "institution", "Curriculum Planning and Coverage", "Centre for Teaching Excellence",
           "Plan a semester and keep delivery on track.", ["curriculum planning", "course design", "time management"],
           "beginner", "online", "free", "2 weeks", "https://institution.example.test/cte/planning"),
    _entry("cat_industry", "institution", "Industry Practice Update for Lecturers", "School of Computing Short Courses",
           "Current tools and practice from industry partners.", ["industry practice", "subject expertise"],
           "intermediate", "in_person", {"amount": 8000, "currency": "KES"}, "1 week",
           "https://institution.example.test/short-courses/industry"),
    _entry("cat_time", "institution", "Time Management for Academic Staff", "Human Resources Training",
           "Plan teaching, research and admin workload.", ["time management"], "beginner", "online", "free",
           "1 day", "https://institution.example.test/hr/time-management"),
]
