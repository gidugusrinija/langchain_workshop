from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = "Srinija"
    age: Optional[int] = 24
    email: EmailStr = "abc@d.com"
    cgpa: float = Field(gt=0.0, lt=10.0, default=9.8,
                        description="Decimal representing the CGPA of the student on a scale of 10")


new_student = {}

student = Student(**new_student)
print(student)
