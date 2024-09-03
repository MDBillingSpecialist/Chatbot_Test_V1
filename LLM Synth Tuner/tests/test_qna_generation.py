import unittest
from unittest.mock import Mock
from src.qna_generation import generate_subtopics, generate_questions, generate_responses

class TestQAGeneration(unittest.TestCase):
    def setUp(self):
        self.mock_client = Mock()

    def test_generate_subtopics(self):
        self.mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Subtopic 1, Subtopic 2"))]
        )
        subtopics = generate_subtopics(self.mock_client, "Sample text")
        self.assertEqual(subtopics, "Subtopic 1, Subtopic 2")

    def test_generate_questions(self):
        self.mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Question 1\nQuestion 2"))]
        )
        questions = generate_questions(self.mock_client, "Sample subtopic")
        self.assertEqual(questions, "Question 1\nQuestion 2")

if __name__ == "__main__":
    unittest.main()
