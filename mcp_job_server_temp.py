"""
mcp_job_server.py — Minimal MCP server for the AI Job Hunt OS.
"""
from mcp.server.fastmcp import FastMCP
import cloudscraper  # Changed from requests
from bs4 import BeautifulSoup

mcp = FastMCP("job-scraper")

@mcp.tool()
def scrape_job_url(url: str) -> dict:
    """
    Scrape a job posting URL and return company, role, and description.
    """
    try:
        # Create a scraper that mimics a real browser to bypass Cloudflare
        scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
        )
        
        resp = scraper.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        company = ""
        role = ""
        description = ""

        og_site = soup.find("meta", property="og:site_name")
        if og_site and og_site.get("content"):
            company = og_site["content"].strip()

        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            role = og_title["content"].strip()
        elif soup.title and soup.title.string:
            role = soup.title.string.strip()

        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
            tag.extract()

        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        description = "\n".join(lines[:150])

        return {
            "company": company,
            "role": role,
            "job_description": description,
            "job_url": url,
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def search_jobs(keywords: str, location: str = "Singapore") -> dict:
    """
    A dummy job search tool to demonstrate LLM tool calling.
    """
    return {
        "query": keywords,
        "location": location,
        "results": [
            {
                "title": f"Senior {keywords.title()}",
                "company": "Example Tech Corp",
                "location": location,
                "url": "https://example.com/jobs/123",
                "salary": "SGD 8,000 - 12,000"
            }
        ],
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")
