from app import db

class Segment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    subtopics = db.relationship('Subtopic', backref='segment', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Segment {self.id}>"

class Subtopic(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    segment_id = db.Column(db.Integer, db.ForeignKey('segment.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    questions = db.relationship('Question', backref='subtopic', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Subtopic {self.id} of Segment {self.segment_id}>"

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    subtopic_id = db.Column(db.Integer, db.ForeignKey('subtopic.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    responses = db.relationship('Response', backref='question', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Question {self.id} of Subtopic {self.subtopic_id}>"

class Response(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('question.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f"<Response {self.id} of Question {self.question_id}>"
