import pytest

from src.abstractor import MockAbstractor
from src.personas import DEFAULT
from src.summarizer_model import SummarizerConfig, TextRankMMRSummarizer
from src.verifier import NERVerifier

SAMPLE_ARTICLE = (
    "The Bank of Canada held its key interest rate steady at 4.5 percent on Wednesday. "
    "Governor Tiff Macklem said the economy is evolving broadly in line with projections. "
    "Inflation has come down significantly from its peak of 8.1 percent last summer. "
    "However, core inflation measures have not shown sustained decline. "
    "The central bank remains prepared to raise rates further if needed. "
    "Economists surveyed before the decision were split on whether the bank would hold or hike. "
    "Financial markets reacted positively to the announcement, with the TSX rising 0.3 percent. "
    "The Canadian dollar weakened slightly against the US dollar after the decision. "
    "Housing markets across Canada continue to show signs of cooling. "
    "The next scheduled rate announcement is set for October 25."
)

SAMPLE_ARTICLE_SHORT = "This is a single sentence article that is long enough to pass the filter."

SAMPLE_ARTICLE_EMPTY = ""


@pytest.fixture
def summarizer():
    return TextRankMMRSummarizer()


@pytest.fixture
def summarizer_config():
    return SummarizerConfig(mmr_lambda=0.75, blend_alpha=0.7, textrank_min_edge=0.1)


@pytest.fixture
def sample_article():
    return SAMPLE_ARTICLE


@pytest.fixture
def sample_article_short():
    return SAMPLE_ARTICLE_SHORT


SAMPLE_SOURCE_WITH_ENTITIES = (
    "Prime Minister Justin Trudeau announced new climate policies in Ottawa on Monday. "
    "The Bank of Canada reported inflation at 3.2 percent for the quarter. "
    "Minister Chrystia Freeland presented the fiscal update in Parliament. "
    "Toronto and Vancouver housing markets showed continued price growth. "
    "The TSX composite index closed at 21,500 points on Friday."
)


@pytest.fixture
def sample_source_with_entities():
    return SAMPLE_SOURCE_WITH_ENTITIES


@pytest.fixture
def mock_abstractor():
    return MockAbstractor()


@pytest.fixture
def verifier():
    return NERVerifier()


@pytest.fixture
def default_persona():
    return DEFAULT
