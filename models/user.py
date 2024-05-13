from app import db

class User(db.Document):
    name = db.StringField(required=True)
    email = db.EmailField(required=True, unique=True)
    password = db.StringField(required=True)

    meta = {'collection': 'users'}  # Optional: Specify the collection name
