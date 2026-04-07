-- =====================================
-- TEACHER MASTER
-- =====================================

CREATE TABLE teacher_master (
    teacher_id SERIAL PRIMARY KEY,
    teacher_name VARCHAR(255),
    email VARCHAR(255),
    password_hash VARCHAR(255),
    created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    isdel INT DEFAULT 0
);


-- =====================================
-- STUDENT MASTER
-- =====================================

CREATE TABLE student_master (
    student_id SERIAL PRIMARY KEY,
    student_name VARCHAR(255),
    roll_number VARCHAR(100),
    class VARCHAR(50),
    created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    isdel INT DEFAULT 0
);


-- =====================================
-- SESSION MASTER
-- =====================================

CREATE TABLE session_master (
    session_id SERIAL PRIMARY KEY,
    session_name VARCHAR(255),
    exam_date DATE,
    teacher_id INT REFERENCES teacher_master(teacher_id),
    total_questions INT,
    created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    isdel INT DEFAULT 0
);


-- =====================================
-- SESSION STUDENT MAPPING
-- =====================================

CREATE TABLE session_student_master (
    session_student_id SERIAL PRIMARY KEY,
    session_id INT REFERENCES session_master(session_id),
    student_id INT REFERENCES student_master(student_id),
    created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    isdel INT DEFAULT 0
);


-- =====================================
-- ANSWER KEY MASTER
-- =====================================

CREATE TABLE answer_key_master (
    answer_key_id SERIAL PRIMARY KEY,
    session_id INT REFERENCES session_master(session_id),
    question_number INT,
    correct_answer VARCHAR(50),
    created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    isdel INT DEFAULT 0
);


-- =====================================
-- ANSWER SHEET MASTER
-- =====================================

CREATE TABLE answer_sheet_master (
    sheet_id SERIAL PRIMARY KEY,
    session_id INT REFERENCES session_master(session_id),
    student_id INT REFERENCES student_master(student_id),
    sheet_image_path TEXT,
    processing_status VARCHAR(50),
    uploaded_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_on TIMESTAMP,
    isdel INT DEFAULT 0
);


-- =====================================
-- ANSWER CROP + OCR RESULT
-- =====================================

CREATE TABLE answer_crop_master (
    crop_id SERIAL PRIMARY KEY,
    sheet_id INT REFERENCES answer_sheet_master(sheet_id),
    question_number INT,
    crop_image_path TEXT,
    ocr_answer VARCHAR(50),
    confidence_score NUMERIC,
    corrected_answer VARCHAR(50),
    is_correct BOOLEAN,
    marks_awarded NUMERIC,
    created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    isdel INT DEFAULT 0
);


-- =====================================
-- FINAL RESULT
-- =====================================

CREATE TABLE result_master (
    result_id SERIAL PRIMARY KEY,
    sheet_id INT REFERENCES answer_sheet_master(sheet_id),
    student_id INT REFERENCES student_master(student_id),
    session_id INT REFERENCES session_master(session_id),
    total_questions INT,
    correct_answers INT,
    total_marks NUMERIC,
    percentage NUMERIC,
    evaluated_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    isdel INT DEFAULT 0
);