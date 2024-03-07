from langchain_community.utilities import SearxSearchWrapper
s = SearxSearchWrapper(searx_host="https://search.bus-hit.me")
s.run("what is a large language model?")