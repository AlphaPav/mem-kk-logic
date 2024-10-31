class LLMBase:
    def __init__(self, model_path=None, api_key=None):
        """
        Initialize a Large Language Model (LLM).
        
        Parameters:
        
        - model_path (str): The file path or URL to the model. Default is None.
        - api_key (str): The API key for querying closed-source models. Default is None.
        
        """

        self.model_path = model_path  # file path or URL that points to the model
        self.api_key = api_key  # API key for accessing LLMs (e.g., ChatGPT)
        self.num_tokens=0
        self.load_model()
    
    def load_model(self):
        pass
    
    def query(self, text):
        """
        Query a model with a given text prompt.
        
        Parameters:
        - text (str): The text prompt to query the model.

        Returns:
        - str: The model's output.
        """
        pass
        
   
