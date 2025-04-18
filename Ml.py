from transformers import pipeline

def create_qa_model():
    """Creates a question-answering model using a pre-trained transformer."""
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return qa_pipeline

def answer_question(qa_pipeline, context, question):
    """Answers a question based on the given context using the QA model."""
    result = qa_pipeline(question=question, context=context)
    return result['answer']

def read_text_file(file_path):
    """Reads a text file and returns its content as a string."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

if __name__ == "__main__":
    # Read text from a file
    file_path = "grph.txt"  # Change this to your file path
    context = read_text_file(file_path)
    
    print("files read ok")
    # Initialize the QA model
    qa_pipeline = create_qa_model()
    


    print("ask question:")
    while True:
        user_input = input(">> ")
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
 
        # Ask a question
        answer = answer_question(qa_pipeline, context, user_input)
    
        print(f"Answer: {answer}")