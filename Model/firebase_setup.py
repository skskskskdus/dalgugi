import firebase_admin
from firebase_admin import credentials,firestore

cred = credentials.Certificate("transitai-firebase-adminsdk-t5n10-5c8949dd9d.json")
firebase_admin.initialize_app(cred)

db = firestore.client()