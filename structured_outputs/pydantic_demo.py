from typing import Optional
from pydantic import BaseModel, EmailStr, Field



class Student(BaseModel):
    name: str
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10)
    
new_student = {'age':25, 'name':"Akash", 'email':'ncack23@gmail.com', 'cgpa':9.9}

student = Student(**new_student)
student_dict = dict(student)

# print(student)
print(student_dict['cgpa'])