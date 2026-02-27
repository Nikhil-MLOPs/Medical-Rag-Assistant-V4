import json
from src.ingestion.ingest import clean_footer, ingest

def test_clean_footer():

    text = """This is some text.
    g a l e e n c y c l o p e d i a"""

    cleaned = clean_footer(text)

    assert cleaned == "This is some text."
    assert "g a l e e n c y c l o p e d i a" not in cleaned


def test_ingest_without_physical_files(mocker):
    
    mock_cfg = mocker.MagicMock() # Create a mock configuration object
    mock_cfg.raw_dir = "fake_raw_path"
    mock_cfg.processed_dir = "fake_out_path"
    mock_cfg.skip_start_pages = 0
    mock_cfg.skip_after_pages = 5
    mocker.patch("src.ingestion.ingest.load_ingestion_config", return_value=mock_cfg) # Mock the configuration loading to return our mock configuration

    
    mocker.patch("src.ingestion.ingest.Path.mkdir") # Mock the mkdir method to do nothing
    fake_pdf_path = mocker.MagicMock() # Create a mock Path object to represent the PDF file
    fake_pdf_path.stem = "virtual_document"

    fake_stat = mocker.MagicMock()
    fake_stat.st_size = 1024 * 1024  # 1 MB
    fake_pdf_path.stat.return_value = fake_stat
    fake_pdf_path.suffix = ".pdf"
    
    mocker.patch("src.ingestion.ingest.Path.glob", return_value=[fake_pdf_path])

    
    mock_doc = mocker.MagicMock() # Create a mock document object
    mock_doc.__len__.return_value = 1
    
    mock_page = mocker.MagicMock()
    mock_page.get_text.return_value = "Content from virtual PDF"
    mock_doc.load_page.return_value = mock_page
    
    mocker.patch("pymupdf.open", return_value=mock_doc)

    
    mock_file = mocker.mock_open() # Mock the open function to capture file writes
    mocker.patch("builtins.open", mock_file)

    
    ingest() # Call the ingest function, which will use the mocked components

    handle = mock_file()
    
    written_call_args = handle.write.call_args_list[0][0][0]
    result = json.loads(written_content := written_call_args.strip())

    assert result["text"] == "Content from virtual PDF"
    assert result["metadata"]["pdf"] == "virtual_document"
    assert result["metadata"]["page"] == 1