import re
import string
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import functools
from performance_optimizer import cached_prediction, performance_monitor

@dataclass
class PIIMatch:
    """Represents a detected PII entity"""
    text: str
    pii_type: str
    start_pos: int
    end_pos: int
    confidence: float
    pattern_name: str

class PIIDetector:
    """Rule-based PII detection engine with high-precision regex patterns"""

    def __init__(self):
        # Arabic linguistic patterns for better contextual detection
        self.arabic_name_indicators = [
            r'الأستاذ|الدكتور|السيد|السيدة|الأخ|الأخت',  # Title indicators
            r'ابن|بنت|أبو|أم',  # Family relation indicators
            r'يدعى|اسمه|اسمها|المسمى|المدعو',  # Name introduction phrases
        ]

        self.arabic_location_indicators = [
            r'مدينة|محافظة|منطقة|حي|شارع|طريق',  # Location type indicators
            r'يقيم في|يسكن في|عنوانه|موقع',  # Address introduction phrases
        ]

        # Compile all regex patterns for better performance
        self.patterns = {
            'PHONE': self._get_phone_patterns(),
            'EMAIL': self._get_email_patterns(),
            'NATIONAL_ID': self._get_national_id_patterns(),
            'IBAN': self._get_iban_patterns(),
            'CREDIT_CARD': self._get_credit_card_patterns(),
            'PASSPORT': self._get_passport_patterns(),
            'LICENSE_PLATE': self._get_license_plate_patterns(),
            'IMEI': self._get_imei_patterns(),
            'IP_ADDRESS': self._get_ip_address_patterns(),
            'LICENSE_NUMBER': self._get_license_number_patterns(),
            'AGE': self._get_age_patterns()
        }

    def _get_phone_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        """Define phone number patterns with high precision"""
        patterns = []

        # Saudi Arabian mobile numbers
        patterns.append((
            "Saudi Mobile (+966)",
            re.compile(r'\+966\s*[5][0-9]\s*\d{3}\s*\d{4}', re.IGNORECASE),
            0.95
        ))

        patterns.append((
            "Saudi Mobile (00966)",
            re.compile(r'00966\s*[5][0-9]\s*\d{3}\s*\d{4}', re.IGNORECASE),
            0.95
        ))

        patterns.append((
            "Saudi Mobile (05)",
            re.compile(r'05\s*[0-9]\s*\d{3}\s*\d{4}(?!\d)', re.IGNORECASE),
            0.90
        ))

        # Jordanian mobile numbers (as mentioned in prompt)
        patterns.append((
            "Jordan Mobile (+962)",
            re.compile(r'\+962\s*[7][7-9]\s*\d{3}\s*\d{4}', re.IGNORECASE),
            0.95
        ))

        patterns.append((
            "Jordan Mobile (00962)",
            re.compile(r'00962\s*[7][7-9]\s*\d{3}\s*\d{4}', re.IGNORECASE),
            0.95
        ))

        patterns.append((
            "Jordan Mobile (07)",
            re.compile(r'07\s*[7-9]\s*\d{3}\s*\d{4}(?!\d)', re.IGNORECASE),
            0.90
        ))

        # UAE mobile numbers
        patterns.append((
            "UAE Mobile (+971)",
            re.compile(r'\+971\s*[5][0-6]\s*\d{3}\s*\d{4}', re.IGNORECASE),
            0.95
        ))

        # Egyptian mobile numbers
        patterns.append((
            "Egypt Mobile (+20)",
            re.compile(r'\+20\s*[1][0-5]\s*\d{4}\s*\d{4}', re.IGNORECASE),
            0.95
        ))

        # Generic international format
        patterns.append((
            "International Phone",
            re.compile(r'\+\d{1,3}\s*\d{1,4}\s*\d{3,4}\s*\d{4}', re.IGNORECASE),
            0.80
        ))

        return patterns

    def _get_email_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        """Define email patterns with high precision"""
        patterns = []

        # Standard email pattern with high precision
        patterns.append((
            "Standard Email",
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', re.IGNORECASE),
            0.95
        ))

        # Arabic domain emails
        patterns.append((
            "Arabic Domain Email",
            re.compile(r'\b[A-Za-z0-9._%+-]+@[\u0600-\u06FF\w.-]+\.[A-Za-z]{2,}\b', re.IGNORECASE),
            0.90
        ))

        return patterns

    def _get_national_id_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        """Define national ID patterns for various Arab countries"""
        patterns = []

        # Saudi National ID (10 digits starting with 1 or 2)
        patterns.append((
            "Saudi National ID",
            re.compile(r'\b[12]\d{9}\b'),
            0.95
        ))

        # UAE Emirates ID (15 digits, format: 784-YYYY-XXXXXXX-X)
        patterns.append((
            "UAE Emirates ID",
            re.compile(r'\b784[-\s]*\d{4}[-\s]*\d{7}[-\s]*\d\b'),
            0.95
        ))

        # Egyptian National ID (14 digits)
        patterns.append((
            "Egyptian National ID",
            re.compile(r'\b[23]\d{13}\b'),
            0.90
        ))

        # Jordanian National ID (10 digits)
        patterns.append((
            "Jordan National ID",
            re.compile(r'\b\d{10}\b'),
            0.80  # Lower confidence due to generic pattern
        ))

        return patterns

    def _get_iban_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        patterns = []

        patterns.extend([
            ("Saudi IBAN", re.compile(r'\bSA\d{2}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\b', re.IGNORECASE), 0.95),
            ("UAE IBAN", re.compile(r'\bAE\d{2}\s*\d{3}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{3}\b', re.IGNORECASE), 0.95),
            ("Egypt IBAN", re.compile(r'\bEG\d{2}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{3}\b', re.IGNORECASE), 0.95),
            ("Jordan IBAN", re.compile(r'\bJO\d{2}\s*[A-Z]{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\b', re.IGNORECASE), 0.95),
            ("Generic IBAN", re.compile(r'\b[A-Z]{2}\d{2}\s*[A-Z0-9]{4}\s*\d{4}\s*\d{4}\s*[A-Z0-9]*\b', re.IGNORECASE), 0.85)
        ])

        return patterns

    def _get_credit_card_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        """Define credit card patterns"""
        patterns = []

        # Visa (starts with 4, 16 digits)
        patterns.append((
            "Visa Card",
            re.compile(r'\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            0.90
        ))

        # MasterCard (starts with 5, 16 digits)
        patterns.append((
            "MasterCard",
            re.compile(r'\b5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            0.90
        ))

        # American Express (starts with 34 or 37, 15 digits)
        patterns.append((
            "American Express",
            re.compile(r'\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b'),
            0.90
        ))

        return patterns

    def _get_passport_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        """Define passport number patterns"""
        patterns = []

        # Saudi passport (letter followed by 7-8 digits)
        patterns.append((
            "Saudi Passport",
            re.compile(r'\b[A-Z]\d{7,8}\b', re.IGNORECASE),
            0.80
        ))

        # Generic passport (various formats)
        patterns.append((
            "Generic Passport",
            re.compile(r'\b[A-Z]{1,2}\d{6,8}\b', re.IGNORECASE),
            0.70
        ))

        return patterns

    def _get_license_plate_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        """Define license plate patterns for Arab countries"""
        patterns = []

        # Saudi license plate (3 Arabic letters + 4 digits)
        patterns.append((
            "Saudi License Plate",
            re.compile(r'[\u0600-\u06FF]{3}\s*\d{4}'),
            0.90
        ))

        # UAE license plate (1-2 letters + 1-5 digits)
        patterns.append((
            "UAE License Plate",
            re.compile(r'\b[A-Z]{1,2}\s*\d{1,5}\b', re.IGNORECASE),
            0.80
        ))

        return patterns

    def _get_imei_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        """Define IMEI number patterns"""
        patterns = [
            # Standard IMEI format: XX-XXXXXX-XXXXXX-X
            (r'\b\d{2}-\d{6}-\d{6}-\d{1}\b', 'IMEI Standard Format', 0.95),
            # Alternative IMEI format: XXXXXXXXXXXXXXX (15 digits)
            (r'\b\d{15}\b', 'IMEI 15-Digit', 0.85),
            # IMEI with spaces: XX XXXXXX XXXXXX X
            (r'\b\d{2}\s\d{6}\s\d{6}\s\d{1}\b', 'IMEI Spaced Format', 0.90),
        ]
        return patterns

    def _get_ip_address_patterns(self) ->  List[Tuple[str, re.Pattern, float]]:
        """Define IP address patterns (IPv4 and IPv6)"""
        patterns = [
            # IPv4 addresses
            (r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b', 'IPv4 Address', 0.90),
            # IPv6 addresses (simplified pattern)
            (r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b', 'IPv6 Full', 0.95),
            # IPv6 compressed format
            (r'\b[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4})*::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4})*)?(?::[0-9a-fA-F]{1,4}){0,7}\b', 'IPv6 Compressed', 0.90),
        ]
        return patterns

    def _get_license_number_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        """Define license/reference number patterns"""
        patterns = [
            # Alphanumeric license format: 78B5R2MVFAHJ48500
            (r'\b[0-9A-Z]{10,20}\b', 'License Alphanumeric', 0.75),
            # Mixed format with numbers and letters
            (r'\b[A-Z]{2,4}\d{4,12}[A-Z]{0,4}\b', 'License Mixed Format', 0.80),
            # Reference numbers (common in Arabic documents)
            (r'\b(?:رقم|رخصة|مرجع)\s*:?\s*([A-Z0-9]{6,15})\b', 'Arabic Reference Number', 0.85),
        ]
        return patterns

    def _get_age_patterns(self) -> List[Tuple[str, re.Pattern, float]]:
        """Define age number patterns in Arabic context"""
        patterns = [
            # Arabic age patterns: "88 عامًا", "من العمر 25"
            (r'(?:من\s+العمر|عمره|عمرها|البالغ)\s+(\d{1,3})\s*(?:عام|سنة)', 'Arabic Age Pattern', 0.90),
            # Simple age in years context
            (r'(\d{1,3})\s+(?:عامًا|سنة|عام)', 'Age Years Arabic', 0.85),
        ]
        return patterns

    def detect_saudi_mobile_numbers(self, text: str) -> List[PIIMatch]:
        """
        Specialized function to detect Saudi Arabian mobile numbers
        Numbers can start with +966, 00966, or 05 and may contain spaces
        """
        matches = []

        # Saudi mobile patterns
        saudi_patterns = [
            (r'\+966\s*[5][0-9]\s*\d{3}\s*\d{4}', "Saudi Mobile (+966)", 0.95),
            (r'00966\s*[5][0-9]\s*\d{3}\s*\d{4}', "Saudi Mobile (00966)", 0.95),
            (r'05\s*[0-9]\s*\d{3}\s*\d{4}(?!\d)', "Saudi Mobile (05)", 0.90),
        ]

        for pattern, name, confidence in saudi_patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            for match in regex.finditer(text):
                matches.append(PIIMatch(
                    text=match.group(),
                    pii_type="PHONE",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence,
                    pattern_name=name
                ))

        return matches

    @cached_prediction("rule_based")
    @performance_monitor.time_function("detect_all_pii")
    def detect_all_pii(self, text: str, min_confidence: float = 0.7) -> List[PIIMatch]:
        """Detect all PII types in the given text"""
        all_matches = []

        for pii_type, patterns in self.patterns.items():
            for pattern_name, regex, confidence in patterns:
                if confidence >= min_confidence:
                    for match in regex.finditer(text):
                        all_matches.append(PIIMatch(
                            text=match.group(),
                            pii_type=pii_type,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=confidence,
                            pattern_name=pattern_name
                        ))

        return self._remove_overlapping_matches(all_matches)

    def _remove_overlapping_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping matches, keeping the one with higher confidence"""
        if not matches:
            return matches

        # Sort by start position
        matches.sort(key=lambda x: x.start_pos)

        filtered_matches = []
        for match in matches:
            # Check if this match overlaps with any already added match
            overlaps = False
            for existing in filtered_matches:
                if (match.start_pos < existing.end_pos and 
                    match.end_pos > existing.start_pos):
                    # There's an overlap
                    if match.confidence > existing.confidence:
                        # Remove the existing match and add this one
                        filtered_matches.remove(existing)
                        filtered_matches.append(match)
                    overlaps = True
                    break

            if not overlaps:
                filtered_matches.append(match)

        return filtered_matches

    def validate_phone_number(self, phone: str) -> bool:
        """Additional validation for phone numbers"""
        # Remove all non-digit characters except +
        clean_phone = re.sub(r'[^\d+]', '', phone)

        # Basic length checks
        if clean_phone.startswith('+966') and len(clean_phone) == 13:
            return True
        elif clean_phone.startswith('00966') and len(clean_phone) == 14:
            return True
        elif clean_phone.startswith('05') and len(clean_phone) == 10:
            return True

        return False

    def validate_iban(self, iban: str) -> bool:
        """Basic IBAN validation using checksum"""
        # Remove spaces and convert to uppercase
        iban = re.sub(r'\s', '', iban).upper()

        # Basic format check
        if not re.match(r'^[A-Z]{2}\d{2}[A-Z0-9]+$', iban):
            return False

        # Move first 4 characters to end and convert letters to numbers
        rearranged = iban[4:] + iban[:4]
        numeric = ''
        for char in rearranged:
            if char.isalpha():
                numeric += str(ord(char) - ord('A') + 10)
            else:
                numeric += char

        # Check if mod 97 equals 1
        return int(numeric) % 97 == 1

    def detect_imei_numbers(self, text: str) -> List[PIIMatch]:
        """Detect IMEI numbers (15-digit device identifiers)"""
        patterns = [
            # Standard IMEI format: XX-XXXXXX-XXXXXX-X
            (r'\b\d{2}-\d{6}-\d{6}-\d{1}\b', 'IMEI Standard Format', 0.95),
            # Alternative IMEI format: XXXXXXXXXXXXXXX (15 digits)
            (r'\b\d{15}\b', 'IMEI 15-Digit', 0.85),
            # IMEI with spaces: XX XXXXXX XXXXXX X
            (r'\b\d{2}\s\d{6}\s\d{6}\s\d{1}\b', 'IMEI Spaced Format', 0.90),
        ]

        matches = []
        for pattern, name, confidence in patterns:
            for match in re.finditer(pattern, text):
                matches.append(PIIMatch(
                    text=match.group(),
                    pii_type="IMEI",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence,
                    pattern_name=name
                ))

        return matches

    def detect_ip_addresses(self, text: str) -> List[PIIMatch]:
        """Detect IP addresses (IPv4 and IPv6)"""
        patterns = [
            # IPv4 addresses
            (r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b', 'IPv4 Address', 0.90),
            # IPv6 addresses (simplified pattern)
            (r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b', 'IPv6 Full', 0.95),
            # IPv6 compressed format
            (r'\b[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4})*::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4})*)?(?::[0-9a-fA-F]{1,4}){0,7}\b', 'IPv6 Compressed', 0.90),
        ]

        matches = []
        for pattern, name, confidence in patterns:
            for match in re.finditer(pattern, text):
                matches.append(PIIMatch(
                    text=match.group(),
                    pii_type="IP_ADDRESS",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence,
                    pattern_name=name
                ))

        return matches

    def detect_license_numbers(self, text: str) -> List[PIIMatch]:
        """Detect license/reference numbers"""
        patterns = [
            # Alphanumeric license format: 78B5R2MVFAHJ48500
            (r'\b[0-9A-Z]{10,20}\b', 'License Alphanumeric', 0.75),
            # Mixed format with numbers and letters
            (r'\b[A-Z]{2,4}\d{4,12}[A-Z]{0,4}\b', 'License Mixed Format', 0.80),
            # Reference numbers (common in Arabic documents)
            (r'\b(?:رقم|رخصة|مرجع)\s*:?\s*([A-Z0-9]{6,15})\b', 'Arabic Reference Number', 0.85),
        ]

        matches = []
        for pattern, name, confidence in patterns:
            for match in re.finditer(pattern, text):
                # For Arabic reference pattern, extract the number part
                if "Arabic" in name and match.groups():
                    number_text = match.group(1)
                    # Find the position of the number within the full match
                    full_match = match.group()
                    number_start = full_match.find(number_text)
                    start_pos = match.start() + number_start
                    end_pos = start_pos + len(number_text)

                    matches.append(PIIMatch(
                        text=number_text,
                        pii_type="LICENSE_NUMBER",
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=confidence,
                        pattern_name=name
                    ))
                else:
                    matches.append(PIIMatch(
                        text=match.group(),
                        pii_type="LICENSE_NUMBER",
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=confidence,
                        pattern_name=name
                    ))

        return matches

    def detect_age_numbers(self, text: str) -> List[PIIMatch]:
        """Detect age numbers in Arabic context"""
        patterns = [
            # Arabic age patterns: "88 عامًا", "من العمر 25"
            (r'(?:من\s+العمر|عمره|عمرها|البالغ)\s+(\d{1,3})\s*(?:عام|سنة)', 'Arabic Age Pattern', 0.90),
            # Simple age in years context
            (r'(\d{1,3})\s+(?:عامًا|سنة|عام)', 'Age Years Arabic', 0.85),
        ]

        matches = []
        for pattern, name, confidence in patterns:
            for match in re.finditer(pattern, text):
                if match.groups():
                    age_number = match.group(1)
                    # Find position of the age number
                    full_match = match.group()
                    number_start = full_match.find(age_number)
                    start_pos = match.start() + number_start
                    end_pos = start_pos + len(age_number)

                    # Validate age range (realistic ages)
                    try:
                        age = int(age_number)
                        if 1 <= age <= 120:  # Reasonable age range
                            matches.append(PIIMatch(
                                text=age_number,
                                pii_type="AGE",
                                start_pos=start_pos,
                                end_pos=end_pos,
                                confidence=confidence,
                                pattern_name=name
                            ))
                    except ValueError:
                        continue

        return matches
    
    def detect_contextual_arabic_pii(self, text: str, min_confidence: float = 0.7) -> List[PIIMatch]:
        """Detect PII entities in Arabic context using linguistic indicators."""
        matches = []

        # Example: Detecting names with context
        for indicator in self.arabic_name_indicators:
            pattern = r'(?:' + indicator + r')\s+([\u0600-\u06FF\s]+)'
            regex = re.compile(pattern, re.IGNORECASE)
            for match in regex.finditer(text):
                name = match.group(1).strip()
                if name:
                    matches.append(PIIMatch(
                        text=name,
                        pii_type="PERSON_NAME",
                        start_pos=match.start(1),
                        end_pos=match.end(1),
                        confidence=0.85,  # Adjust confidence as needed
                        pattern_name="Arabic Name with Context"
                    ))

        # Example: Detecting locations with context
        for indicator in self.arabic_location_indicators:
            pattern = r'(?:' + indicator + r')\s+([\u0600-\u06FF\s]+)'
            regex = re.compile(pattern, re.IGNORECASE)
            for match in regex.finditer(text):
                location = match.group(1).strip()
                if location:
                    matches.append(PIIMatch(
                        text=location,
                        pii_type="LOCATION",
                        start_pos=match.start(1),
                        end_pos=match.end(1),
                        confidence=0.80,  # Adjust confidence as needed
                        pattern_name="Arabic Location with Context"
                    ))

        return matches

def test_pii_detector():
    """Test function to demonstrate the PII detector capabilities"""
    detector = PIIDetector()

    # Test text with various PII types
    test_text = """
    اتصل بي على رقم 0501234567 أو +966501234567
    إيميلي هو ahmed@example.com
    رقم الهوية الوطنية: 1234567890
    الآيبان: SA12 3456 7890 1234 5678 9012 3456
    رقم البطاقة الائتمانية: 4111-1111-1111-1111
    رقم جواز السفر: A1234567

    You can also reach me at john.doe@company.org
    Phone: +962 77 123 4567
    IBAN: JO94 CBJO 0010 0000 0000 0131 0003
    IMEI: 12-345678-901234-5
    IP Address: 192.168.1.1
    License Number: A1234567890
    Age: عمري 25 عاما

    """

    print("Testing Saudi Mobile Number Detection:")
    print("=" * 50)
    saudi_mobiles = detector.detect_saudi_mobile_numbers(test_text)
    for match in saudi_mobiles:
        print(f"Found: {match.text} ({match.pattern_name}) - Confidence: {match.confidence}")

    print("\nTesting All PII Detection:")
    print("=" * 50)
    all_pii = detector.detect_all_pii(test_text, min_confidence=0.8)
    for match in all_pii:
        print(f"{match.pii_type}: {match.text} ({match.pattern_name}) - Confidence: {match.confidence}")

if __name__ == "__main__":
    test_pii_detector()