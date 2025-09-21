from typing import TypedDict

class Person(TypedDict):
    
    name: str
    age: int
    
new_person: Person = { "name": "Akash", "age":25}

print(new_person)
    