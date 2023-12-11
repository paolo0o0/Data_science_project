FROM python:3.9-slim

COPY . C:\Users\Павел\Desktop\М.Тех_ТЗ_DS
WORKDIR C:\Users\Павел\Desktop\root_directory

RUN apt-get update 

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/paolo0o0/Data_science_project

RUN pip install streamlit
RUN pip install numpy
RUN pip install pandas
RUN pip install statsmodels
RUN pip install plotly

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "https://github.com/paolo0o0/Data_science_project/blob/main/data_science.py"]