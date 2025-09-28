from dotenv import load_dotenv
load_dotenv()

from crewai_tools import CSVSearchTool , FileReadTool
from crewai import Agent, Crew, Task

config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama3.1:latest",
                base_url= "http://localhost:11434"
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            )
        ),
        embedder=dict(
                provider="ollama", # or google, openai, anthropic, 
                config=dict(
                    model="mxbai-embed-large:latest", 
                    base_url= "http://localhost:11434"
                )
            )
        )

#csv_tool = CSVSearchTool(csv='arhivaLoto.csv', config = config) 
csv_tool = FileReadTool(file_path='arhivaLoto2.csv')          

statistical_agent = Agent(
    role="Data Scientist Agent",
    goal=f"""Predict next 6 number""",
    backstory="""
        You are an experienced data scientist who try to find patterns on sets of data       
        Try to analyze the numbers in given csv file and count each number between 1 and 49 for how many times appear in whole dataset
        Try to get the most correlated and frequent numbers by how many time appears
    """,
    verbose=True,
    tools= [csv_tool],
    llm ="ollama/llama3.1:latest",
    memory=True,
    allow_delegation=False
)

statistical_task = Task(
    description=f"""Find the frequency of each number between 1 and 49 and display top 6 numbers by frequency
                    Each row represent the 6 randomly extracted numbers on a week
                    First row represent columns name and is of type string could be ignored in calculations
                    Identify trends or patterns or correlation in the frequency of numbers  
                 """,
    expected_output =f"""Return only the top 6 most frequent numbers counted by how many times each number appears in dataset 
                         Example: Most frequent 6 numbers are -> 23 appears 50 times 
                                                                 12 appears 49 times 
                                                                  7 appears 47 times 
                      """ , 
    agent=statistical_agent
)
crew = Crew(
    agents=[statistical_agent],
    tasks=[statistical_task],
    memory=False,
    output_log_file= "numberPicker.txt"
)

result = crew.kickoff()
print(result)