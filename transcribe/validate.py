"""Checking a MusicXML file the way a notation program will.

Well-formed XML is a weak test. A file can parse perfectly and still be
rejected as corrupt, because MusicXML carries rules that XML syntax knows
nothing about: a voice number below 1, a bar holding more music than its time
signature allows, a time cursor driven negative by too many `backup` elements.

Every one of those was shipped during this project's development and found only
when a notation program refused the file — twice from our own writer, once from
the recognition engine upstream. These are the checks that would have caught
them at the point of writing.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    measure: str
    detail: str

    def __str__(self) -> str:
        return f"measure {self.measure}: {self.detail}"


def _cursor_bounds(measure: ET.Element) -> tuple[int, int]:
    """Lowest and highest the time cursor reaches while reading a measure."""
    position = low = high = 0
    for child in measure:
        if child.tag == "note":
            if child.find("chord") is None and child.find("grace") is None:
                position += int(child.findtext("duration") or 0)
        elif child.tag == "forward":
            position += int(child.findtext("duration") or 0)
        elif child.tag == "backup":
            position -= int(child.findtext("duration") or 0)
        low, high = min(low, position), max(high, position)
    return low, high


def validate(path: str) -> list[Problem]:
    """Problems a notation program would report, in reading order."""
    root = ET.parse(path).getroot()
    problems: list[Problem] = []

    listed = {part.get("id") for part in root.findall("part-list/score-part")}
    used = {part.get("id") for part in root.findall("part")}
    if listed != used:
        problems.append(Problem("-", f"part-list {sorted(listed)} does not match parts {sorted(used)}"))

    for part in root.findall("part"):
        divisions = beats = beat_type = 0
        staves = 1

        for measure in part.findall("measure"):
            number = measure.get("number") or "?"

            for attributes in measure.findall("attributes"):
                if attributes.findtext("divisions"):
                    divisions = int(attributes.findtext("divisions"))
                if attributes.findtext("staves"):
                    staves = int(attributes.findtext("staves"))
                found_beats = attributes.findtext("time/beats")
                found_type = attributes.findtext("time/beat-type")
                if found_beats and found_type:
                    beats, beat_type = int(found_beats), int(found_type)
                if staves > 1:
                    for clef in attributes.findall("clef"):
                        if clef.get("number") is None:
                            problems.append(
                                Problem(number, f"a clef has no number in a {staves}-staff part")
                            )

            for element in measure.findall("note"):
                voice = element.findtext("voice")
                if voice is not None and (not voice.strip().isdigit() or int(voice) < 1):
                    problems.append(Problem(number, f"<voice>{voice}</voice> must be 1 or greater"))
                    break
                staff = element.findtext("staff")
                if staff is not None and not 1 <= int(staff) <= staves:
                    problems.append(Problem(number, f"<staff>{staff}</staff> outside 1..{staves}"))
                    break
                if element.find("chord") is not None and list(element)[0].tag != "chord":
                    problems.append(Problem(number, "<chord/> must be the first child of its note"))
                    break

            low, high = _cursor_bounds(measure)
            if low < 0:
                problems.append(Problem(number, f"time cursor goes negative ({low})"))
            elif divisions and beats and beat_type:
                full = divisions * beats * 4 // beat_type
                if high > full:
                    problems.append(Problem(number, f"holds more than the bar allows ({high} > {full})"))

    return problems
