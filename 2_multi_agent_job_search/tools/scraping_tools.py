"""
tools/scraping_tools.py
────────────────────────
CrewAI-compatible tools for scraping job listings from multiple platforms.
Each tool is a @tool-decorated function that agents can call autonomously.

Scraping strategy:
 - LinkedIn / Wellfound: playwright (JS-heavy pages)
 - Indeed: httpx + BeautifulSoup (mostly static)
 - Company pages: generic playwright fallback
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from typing import Any

import httpx
from bs4 import BeautifulSoup
from crewai.tools import tool
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings


def _random_delay() -> None:
    delay = random.uniform(settings.scrape_delay_min, settings.scrape_delay_max)
    time.sleep(delay)


def _job_hash(title: str, company: str) -> str:
    return hashlib.md5(f"{title.lower().strip()}{company.lower().strip()}".encode()).hexdigest()


# ── Indeed ────────────────────────────────────────────────────


@tool("search_indeed_jobs")
def search_indeed_jobs(keywords: str, location: str = "Remote", max_results: int = 20) -> str:
    """
    Search Indeed for job listings matching keywords and location.
    Returns a JSON string with a list of job dictionaries.

    Args:
        keywords:    Job title / skills to search for (e.g. "Python Engineer")
        location:    City, state, or "Remote"
        max_results: Maximum number of jobs to return (default 20)
    """
    results: list[dict[str, Any]] = []

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        params = {
            "q": keywords,
            "l": location,
            "limit": min(max_results, 50),
            "fromage": 14,  # last 14 days
        }

        with httpx.Client(timeout=settings.request_timeout, headers=headers, follow_redirects=True) as client:
            response = client.get("https://www.indeed.com/jobs", params=params)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        cards = soup.select("div.job_seen_beacon") or soup.select("[data-jk]")

        for card in cards[: max_results]:
            try:
                title_el = card.select_one("h2.jobTitle span") or card.select_one("h2 a span")
                company_el = card.select_one("[data-testid='company-name']") or card.select_one(".companyName")
                location_el = card.select_one("[data-testid='text-location']") or card.select_one(".companyLocation")
                salary_el = card.select_one("[data-testid='attribute_snippet_testid']")
                link_el = card.select_one("h2 a")

                title = title_el.get_text(strip=True) if title_el else "Unknown"
                company = company_el.get_text(strip=True) if company_el else "Unknown"
                job_location = location_el.get_text(strip=True) if location_el else location
                salary = salary_el.get_text(strip=True) if salary_el else None
                url = "https://indeed.com" + link_el["href"] if link_el and link_el.get("href") else None

                results.append(
                    {
                        "title": title,
                        "company": company,
                        "location": job_location,
                        "salary": salary,
                        "application_url": url,
                        "source": "indeed",
                        "external_id": _job_hash(title, company),
                        "is_remote": "remote" in job_location.lower(),
                    }
                )
            except Exception as e:
                logger.debug(f"Skipping card: {e}")
                continue

        _random_delay()

    except Exception as e:
        logger.error(f"Indeed scraping failed: {e}")
        return json.dumps({"error": str(e), "jobs": []})

    logger.info(f"Indeed: found {len(results)} jobs for '{keywords}' in '{location}'")
    return json.dumps({"jobs": results, "count": len(results), "source": "indeed"})


# ── LinkedIn (async playwright) ──────────────────────────────


@tool("search_linkedin_jobs")
def search_linkedin_jobs(keywords: str, location: str = "Remote", max_results: int = 20) -> str:
    """
    Search LinkedIn Jobs for listings matching keywords and location.
    Uses a headless browser to bypass JavaScript rendering.
    Returns JSON with list of job dictionaries.

    Args:
        keywords:    e.g. "Machine Learning Engineer"
        location:    e.g. "San Francisco, CA" or "Remote"
        max_results: cap on results (default 20)
    """
    return asyncio.run(_async_linkedin(keywords, location, max_results))


async def _async_linkedin(keywords: str, location: str, max_results: int) -> str:
    results: list[dict[str, Any]] = []
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=settings.use_headless_browser)
            page = await browser.new_page()
            await page.set_extra_http_headers(
                {"Accept-Language": "en-US,en;q=0.9"}
            )

            url = (
                f"https://www.linkedin.com/jobs/search/?keywords={keywords.replace(' ', '%20')}"
                f"&location={location.replace(' ', '%20')}&f_TPR=r604800"
            )
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(3000)

            cards = await page.query_selector_all(".base-card")
            for card in cards[:max_results]:
                try:
                    title = await (await card.query_selector(".base-search-card__title")).inner_text()
                    company = await (await card.query_selector(".base-search-card__subtitle")).inner_text()
                    loc_el = await card.query_selector(".job-search-card__location")
                    loc = await loc_el.inner_text() if loc_el else location
                    link_el = await card.query_selector("a.base-card__full-link")
                    link = await link_el.get_attribute("href") if link_el else None

                    results.append(
                        {
                            "title": title.strip(),
                            "company": company.strip(),
                            "location": loc.strip(),
                            "application_url": link,
                            "source": "linkedin",
                            "external_id": _job_hash(title, company),
                            "is_remote": "remote" in loc.lower(),
                        }
                    )
                except Exception as e:
                    logger.debug(f"LinkedIn card skip: {e}")

            await browser.close()

    except Exception as e:
        logger.error(f"LinkedIn scraping failed: {e}")
        return json.dumps({"error": str(e), "jobs": []})

    logger.info(f"LinkedIn: found {len(results)} jobs for '{keywords}'")
    return json.dumps({"jobs": results, "count": len(results), "source": "linkedin"})


# ── Wellfound (AngelList) ────────────────────────────────────


@tool("search_wellfound_jobs")
def search_wellfound_jobs(keywords: str, max_results: int = 15) -> str:
    """
    Search Wellfound (AngelList) for startup job listings.
    Returns JSON with list of job dictionaries.

    Args:
        keywords:    Role or skills to search (e.g. "Backend Engineer")
        max_results: Maximum results (default 15)
    """
    return asyncio.run(_async_wellfound(keywords, max_results))


async def _async_wellfound(keywords: str, max_results: int) -> str:
    results: list[dict[str, Any]] = []
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=settings.use_headless_browser)
            page = await browser.new_page()
            slug = keywords.lower().replace(" ", "-")
            await page.goto(
                f"https://wellfound.com/role/r/{slug}",
                wait_until="domcontentloaded",
                timeout=30_000,
            )
            await page.wait_for_timeout(3000)

            cards = await page.query_selector_all("[class*='JobListing']")
            for card in cards[:max_results]:
                try:
                    title_el = await card.query_selector("a[class*='title']")
                    company_el = await card.query_selector("[class*='companyName']")
                    salary_el = await card.query_selector("[class*='compensation']")
                    skills_els = await card.query_selector_all("[class*='tag']")

                    title = await title_el.inner_text() if title_el else "Unknown"
                    company = await company_el.inner_text() if company_el else "Unknown"
                    salary = await salary_el.inner_text() if salary_el else None
                    skills = [await el.inner_text() for el in skills_els]
                    link = await title_el.get_attribute("href") if title_el else None

                    results.append(
                        {
                            "title": title.strip(),
                            "company": company.strip(),
                            "location": "Remote",
                            "salary": salary,
                            "required_skills": skills,
                            "application_url": f"https://wellfound.com{link}" if link else None,
                            "source": "wellfound",
                            "external_id": _job_hash(title, company),
                            "is_remote": True,
                        }
                    )
                except Exception as e:
                    logger.debug(f"Wellfound card skip: {e}")

            await browser.close()

    except Exception as e:
        logger.error(f"Wellfound scraping failed: {e}")
        return json.dumps({"error": str(e), "jobs": []})

    logger.info(f"Wellfound: found {len(results)} jobs for '{keywords}'")
    return json.dumps({"jobs": results, "count": len(results), "source": "wellfound"})


# ── Generic company career page ──────────────────────────────


@tool("scrape_company_jobs")
def scrape_company_jobs(career_page_url: str, company_name: str) -> str:
    """
    Scrape job listings from a specific company's careers page.

    Args:
        career_page_url: Direct URL to the careers/jobs page
        company_name:    Company name for labelling results
    """
    return asyncio.run(_async_company(career_page_url, company_name))


async def _async_company(url: str, company: str) -> str:
    results: list[dict[str, Any]] = []
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=settings.use_headless_browser)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=30_000)

            content = await page.content()
            soup = BeautifulSoup(content, "lxml")

            # Generic heuristic: find anchor tags containing job-related keywords
            for a in soup.find_all("a", href=True):
                text = a.get_text(strip=True)
                href = a["href"]
                if any(kw in text.lower() for kw in ["engineer", "developer", "analyst", "manager", "designer"]):
                    full_url = href if href.startswith("http") else url.rstrip("/") + "/" + href.lstrip("/")
                    results.append(
                        {
                            "title": text[:200],
                            "company": company,
                            "application_url": full_url,
                            "source": "company_site",
                            "external_id": _job_hash(text, company),
                            "is_remote": "remote" in text.lower(),
                        }
                    )

            await browser.close()

    except Exception as e:
        logger.error(f"Company page scraping failed ({url}): {e}")
        return json.dumps({"error": str(e), "jobs": []})

    logger.info(f"Company page ({company}): found {len(results)} job links")
    return json.dumps({"jobs": results[:settings.max_jobs_per_source], "count": len(results)})
