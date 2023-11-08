import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="gptdata_final.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You will help the user generate stable diffusion image prompts. Given an image description, generate a stable diffusion prompt by following ALL of the instructions below.

1/ Use this formula to generate the Stable Diffusion prompt: [Subject], [Adjective], [Doing Action], in [location/setting], [Creative Lighting Style], detailed, anime style, 90s anime style, trending on ArtStation, in the style of [Famous Artist 1] and [Famous Artist 2]

2/ Be sure to keep the [Doing action] and "in [location/setting]"  sections separated by commas, do not merge the two. 

3/ Do not include any information about the location/setting inside the [Doing Action] section, only describe what is being done.

4/ Do not include commas within a section. Commas should only be used to separate one section from another.

Below is an image description:
{message}

Here is a list of previously-generated stable diffusion prompts for other image descriptions:
{best_practice}

Please write the best stable diffusin prompt for this image description:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

message = "Stable diffusion prompt for a brave samurai"
response = generate_response(message)
print(response)


# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Stable diffusion prompt generator", page_icon=":bird:")

    st.header("Stable diffusion prompt generator :bird:")
    message = st.text_area("Image description")

    if message:
        st.write("Generating best stable diffusion prompt...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()
