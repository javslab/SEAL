FROM python:3.6

COPY . /src

WORKDIR /src

RUN /bin/bash -c 'pip install virtualenv && \
    virtualenv .venv && \
    source .venv/bin/activate && \
    pip install -r requirements.txt && \
    pip install sphinx_rtd_theme'

CMD /bin/bash -c 'source .venv/bin/activate && \
    coverage run -m pytest seal && \
    coverage report -m -i'
