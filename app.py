import re
from datetime import datetime
from dateutil.parser import parse

def extract_and_format_date(text_line):
    """
    Extracts date patterns from raw OCR strings across all packaging formats:
    1. Text months: '10 JAN 2027', 'EXP 04-SEP-26', 'OCT 2027'
    2. Separated numeric: '10/01/2027', '04-09-17', '10.01.27', '10 01 2027'
    3. Concatenated numeric: '040917', '20270110'
    """
    clean_line = text_line.strip()

    # Pattern 1: Dates with named months (e.g., "10 JAN 2027", "EXP 04-SEP-26", "BEST BEFORE OCT 2027")
    text_month_match = re.search(
        r'(\b\d{1,2}[\s/.-]?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s/.-]?\d{2,4}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s/.-]?\d{2,4}\b)',
        clean_line, re.I
    )
    if text_month_match:
        target_str = text_month_match.group(0)
        return convert_to_status_wording(target_str)

    # Pattern 2: Standard separated dates with '/', '-', '.', or spaces (e.g., "10/01/2027", "04-09-17", "10 01 2027")
    separated_match = re.search(
        r'(\b\d{1,2}[\s/.-]\d{1,2}[\s/.-]\d{2,4}\b|\b\d{4}[\s/.-]\d{1,2}[\s/.-]\d{1,2}\b)',
        clean_line
    )
    if separated_match:
        target_str = separated_match.group(0)
        return convert_to_status_wording(target_str)

    # Pattern 3: Concatenated digits following EXP/MFG/BB keywords or standalone (e.g., "EXP:040917", "20270110")
    concat_match = re.search(r'(?:exp|bb|mfg|use\s*by)?[\s:]*(\b\d{6}\b|\b\d{8}\b)', clean_line, re.I)
    if concat_match:
        digits = concat_match.group(1)
        if len(digits) == 6:
            # Converts '040917' -> '04/09/2017'
            target_str = f"{digits[:2]}/{digits[2:4]}/20{digits[4:]}"
        elif len(digits) == 8:
            if digits.startswith("20"):  # YYYYMMDD format (e.g., 20270110)
                target_str = f"{digits[:4]}/{digits[4:6]}/{digits[6:]}"
            else:  # DDMMYYYY format (e.g., 10012027)
                target_str = f"{digits[:2]}/{digits[2:4]}/{digits[4:]}"
        return convert_to_status_wording(target_str)

    return clean_line


def convert_to_status_wording(date_str):
    """Converts extracted date string into natural wording and remaining days."""
    try:
        dt = parse(date_str, fuzzy=True)
        wording = dt.strftime("%B %d, %Y")

        today = datetime.now()
        days_left = (dt - today).days

        if days_left < 0:
            status = f"Expired {abs(days_left)} days ago"
        elif days_left == 0:
            status = "Expires today!"
        else:
            status = f"Valid for {days_left} more days"

        return f"{wording} ({status})"
    except Exception:
        return date_str
