import json
import logging
import random
import re
import shutil
import struct
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element

import mobase
import psutil
import vdf  # type: ignore
from PyQt6.QtCore import QDir, QFileInfo, QStandardPaths, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..basic_game import BasicGame, BasicGameSaveGame
from ..steam_utils import find_games, find_steam_path, parse_library_info

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@dataclass
class regex_json_data:
    regex: str
    identifier: list[str]
    file_name: str


class util:
    def __init__(self):
        pass

    @staticmethod
    def try_read_text(file_path: Path) -> str:
        encodings_to_try = ["gbk", "utf-8", "iso-8859-1"]
        for encoding in encodings_to_try:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to decode {file_path} with known encodings.")

    @staticmethod
    def smerge_jsons(
        paths: list[Path], identifier: list[str]
    ) -> dict[str, list[dict[str, Any] | list[str]]]:
        result: dict[str, dict[tuple[str, ...], Any]] = defaultdict(dict)
        for path in paths:
            result_temp: dict[str, dict[tuple[str, ...], Any]] = defaultdict(dict)
            try:
                dict1: dict[str, list[dict[str, Any] | list[str]] | dict[str, str]] = (
                    json.loads(util.try_read_text(path))
                )
                for p_key, p_list in dict1.items():
                    if isinstance(p_list, list):
                        others: list[Any] = []
                        s_dict: dict[tuple[str, ...], Any] = {}
                        for p_value in p_list:
                            if isinstance(p_value, dict):
                                idf = set(identifier) & set(p_value.keys())
                                if idf:
                                    s_key: tuple[str, ...] = tuple(
                                        p_value[i] for i in idf
                                    )
                                    s_dict[s_key] = p_value
                                else:
                                    others.append(p_value)
                            else:
                                others.append(p_value)
                        result_temp[p_key].update(s_dict)
                        result_temp[p_key].update(
                            {(f"other_{index}",): i for index, i in enumerate(others)}
                        )
                    else:
                        raise ValueError(
                            f"Unexpected data type in {path} for {p_key}: {type(p_list)}"
                        )
            except Exception as e:
                logger.error(f"Error in {path}: {e}")
                continue
            result.update(result_temp)

        output: dict[str, list[dict[str, Any] | list[str]]] = {
            k: list(v.values()) for k, v in result.items()
        }
        return output

    @staticmethod
    def scopy_mod(scr: str | Path, dst: str | Path) -> str:  # type: ignore
        scr = str(scr)
        dst = str(dst)
        if not Path(scr).exists():
            return ""
        if Path(dst).exists():
            return util.scopy_mod(scr, f"{dst}_copy")  # type: ignore
        else:
            shutil.copytree(scr, dst)
            return dst

    @staticmethod
    def is_number_and_percent_str(s: str):
        for char in s:
            if not (char.isdigit() or char == "%"):
                return False
        return True

    @staticmethod
    def real_int(string: str):
        try:
            int(string)
            return True
        except Exception:
            return False

    @staticmethod
    def real_float(string: str):
        try:
            float(string)
            return True
        except Exception:
            return False


class darkest:
    class obj(str):
        pass

    def __init__(
        self,
        data: dict[
            str,
            list[dict[str, list[str | bool | int | float] | str | bool | int | float]],
        ],
        strict: bool = True,
    ):
        self.data = data
        self.__strict: bool = strict

    def __str__(self) -> str:
        if self.__strict:
            content = ""
            for p_key, p_values in self.data.items():
                for p_value in p_values:
                    content += f"{p_key}: "
                    for s_key, s_values in p_value.items():
                        content += f".{s_key} "
                        if isinstance(s_values, list):
                            for s_value in s_values:
                                content += f"{s_value} "
                        else:
                            content += f"{s_values} "
                    content += "\n"
            return content
        else:
            content = ""
            for p_key, p_values in self.data.items():
                for p_value in p_values:
                    content += f"{p_key}: "
                    for s_key, s_values in p_value.items():
                        content += f".{s_key} "
                        if not isinstance(s_values, list):
                            s_values = [s_values]
                        for s_value in s_values:
                            if isinstance(s_value, darkest.obj):
                                content += f"{s_value} "
                            elif isinstance(s_value, str):
                                content += f'"{s_value}" '
                            elif isinstance(s_value, bool):
                                content += f"{s_value} ".lower()
                            elif isinstance(s_value, int):
                                content += f"{s_value} "
                            else:
                                content += f"{s_value} "
                    content += "\n"
            return content

    @staticmethod
    def paser(s: str, strict: bool = True):
        # logger.debug(f"pasering: {s}")
        pattern = r'(?:"[^"]*"|\S+)'
        result: dict[
            str,
            list[dict[str, list[str | bool | int | float] | str | bool | int | float]],
        ] = defaultdict(list)
        current_p_key: str = ""
        current_p_value: dict[
            str, list[str | bool | int | float] | str | bool | int | float
        ] = {}
        current_s_key: str = ""
        current_s_value: list[str | bool | int | float] = []
        content: str = re.sub(r"//.*\n", "", s)
        sections: list[str] = re.findall(pattern, content)
        for section in sections:
            if section[-1] == ":":
                if current_p_key:
                    if strict:
                        current_p_value.update({current_s_key: current_s_value})
                    elif len(current_s_value) == 1:
                        current_p_value.update({current_s_key: current_s_value[0]})
                    else:
                        current_p_value.update({current_s_key: current_s_value})
                    result[current_p_key].append(current_p_value)
                current_p_key = section[:-1]
                current_p_value = {}
                current_s_key = ""
                current_s_value = []
            elif section[0] == "." and not util.is_number_and_percent_str(section[1:]):
                if current_s_key:
                    if len(current_s_value) == 1:
                        current_p_value.update({current_s_key: current_s_value[0]})
                    else:
                        current_p_value.update({current_s_key: current_s_value})
                current_s_key = section[1:]
                current_s_value = []
            else:
                if strict:
                    current_s_value.append(section)
                else:
                    if (section[0] == '"' and section[-1] == '"') or (
                        section[0] == "'" and section[-1] == "'"
                    ):
                        current_s_value.append(section[1:-1])
                    elif section.lower() == "true":
                        current_s_value.append(True)
                    elif section.lower() == "false":
                        current_s_value.append(False)
                    elif util.real_int(section):
                        current_s_value.append(int(section))
                    elif util.real_float(section):
                        current_s_value.append(float(section))
                    else:
                        current_s_value.append(darkest.obj(section))
        if current_p_key:
            if strict:
                current_p_value.update({current_s_key: current_s_value})
            elif len(current_s_value) == 1:
                current_p_value.update({current_s_key: current_s_value[0]})
            else:
                current_p_value.update({current_s_key: current_s_value})
            result[current_p_key].append(current_p_value)
        return darkest(result, strict)


class Meta1node:
    def __init__(self, id: int):
        self.id: int = id
        self.key: str = f"{id}"
        self.value: dict[str, Any] = {}
        self.children: dict[int, Meta1node] = {}
        self.parent: Meta1node

    def set_key(self, key: str):
        self.key = key

    def add_value(self, value: Dict[str, Any]):
        self.value.update(value)

    def to_dict(self) -> Dict[str, Any]:
        result = self.value.copy()
        for node in self.children.values():
            result.update(node.to_dict())
        return {self.key: result}


class Meta1nodeManager:
    def __init__(self) -> None:
        self._nodes: Dict[int, Meta1node] = {}

    def get_node(self, id: int) -> Meta1node | None:
        return self._nodes.get(id)

    def add_node(self, id: int):
        self._nodes[id] = Meta1node(id)
        return self._nodes[id]

    def add_child(self, parent: Meta1node, child: Meta1node):
        parent.children[child.id] = child
        child.parent = parent

    def to_dict(self) -> Dict[str, Any]:
        return self._nodes[0].to_dict()

    def parse_raw(self, raw_data: BytesIO, numMeta1Entries: int):
        for x in range(numMeta1Entries):
            parent_id = int.from_bytes(raw_data.read(4), "little")
            raw_data.seek(12, 1)
            child_node = self.add_node(x)
            if parent_id in self._nodes:
                parent = self.get_node(parent_id)
                child = child_node
                if parent is not None:
                    self.add_child(parent, child)


class persist:
    def __init__(self, file: str | Path):
        self.file = Path(file)

    TYPE_FLOAT = [
        "current_hp",
        "m_Stress",
        "actor",
        "buff_group",
        "amount",
        "chapters",
        "percent",
        "non_rolled_additional_chances",
        "chance",
    ]

    TYPE_INTVECTOR = [
        "read_page_indexes",
        "raid_read_page_indexes",
        "raid_unread_page_indexes",  # journal.json
        "dungeons_unlocked",
        "played_video_list",  # game_knowledge.json
        "trinket_retention_ids",  # quest.json
        "last_party_guids",
        "dungeon_history",
        "buff_group_guids",  # roster.json
        "result_event_history",  # town_event.json
        "dead_hero_entries",  # town_event.json
        "additional_mash_disabled_infestation_monster_class_ids",  # campaign_mash.json
        "mash",
        "valid_additional_mash_entry_indexes",  # raid.json
        "party",
        "heroes",  # raid.json
        "skill_cooldown_keys",  # raid.json
        "skill_cooldown_values",
        "bufferedSpawningSlotsAvailable",  # raid.json
        "curioGroups",
        "curios",  # raid.json
        "curioGroups",
        "curio_table_entries",  # raid.json
        "raid_finish_quirk_monster_class_ids",  # raid.json
        "narration_audio_event_queue_tags",  # loading_screen.json
        "dispatched_events",  # tutorial.json
        "backer_heroes",
        "combat_skills",
        "backer_heroes",
        "camping_skills",
        "backer_heroes",
        "quirks",
    ]

    TYPE_STRINGVECTOR = [
        "goal_ids",  # quest.json
        "roaming_dungeon_2_ids",
        "s",  # campaign_mash.json
        "quirk_group",  # campaign_log.json
        "backgroundNames",  # raid.json
        "backgroundGroups",
        "backgrounds",  # raid.json
        "backgroundGroups",
        "background_table_entries",
    ]

    TYPE_FLOATARRAY = [
        "map",
        "bounds",
        "areas",
        "bounds",
        "areas",
        "tiles",
        "mappos",
        "areas",
        "tiles",
        "sidepos",
    ]

    TYPE_TWOINT = ["killRange"]

    @staticmethod
    def isA(type: List[str], name: str):
        return name in type

    @staticmethod
    def parseFloatArray(
        name: str, valueBytes: bytes, alignment_skip: int, aligned_size: int
    ) -> list[str] | Literal[False]:
        if persist.isA(persist.TYPE_FLOATARRAY, name):
            floats = valueBytes[alignment_skip : alignment_skip + aligned_size]
            buffer = BytesIO(floats)
            sb: List[str] = []
            while buffer.tell() < len(floats):
                f = struct.unpack_from("<f", buffer.read(4))[0]
                sb.append(str(f))
            return sb
        return False

    @staticmethod
    def parseIntVector(
        name: str, valueBytes: bytes, alignment_skip: int, aligned_size: int
    ) -> list[str] | Literal[False]:
        if persist.isA(persist.TYPE_INTVECTOR, name):
            tempArr = valueBytes[alignment_skip : alignment_skip + 4]
            arrLen = struct.unpack("<I", tempArr)[0]
            if aligned_size == (arrLen + 1) * 4:
                tempArr2 = valueBytes[
                    alignment_skip + 4 : alignment_skip + (arrLen + 1) * 4
                ]
                buffer = BytesIO(tempArr2)
                sb: List[str] = []
                for _i in range(arrLen):
                    tempInt = struct.unpack_from("<I", buffer.read(4))[0]
                    sb.append(str(tempInt))
                return sb
        return False

    @staticmethod
    def parseStringVector(
        name: str, valueBytes: bytes, alignment_skip: int, aligned_size: int
    ) -> list[str] | Literal[False]:
        if persist.isA(persist.TYPE_STRINGVECTOR, name):
            tempArr = valueBytes[alignment_skip : alignment_skip + 4]
            arrLen = int.from_bytes(tempArr, "little")
            strings = valueBytes[alignment_skip + 4 : alignment_skip + aligned_size]
            bf = BytesIO(strings)
            sb: List[str] = []
            for _i in range(arrLen):
                strlen = struct.unpack_from("<I", bf.read(4))[0]
                tempArr2 = valueBytes[
                    alignment_skip + 4 + bf.tell() : alignment_skip
                    + 4
                    + bf.tell()
                    + strlen
                    - 1
                ]
                sb.append(tempArr2.decode("utf-8"))
                bf.seek(bf.tell() + strlen)
            return sb
        return False

    @staticmethod
    def parseFloat(
        name: str, valueBytes: bytes, alignment_skip: int, aligned_size: int
    ) -> str | Literal[False]:
        if persist.isA(persist.TYPE_FLOAT, name):
            if aligned_size == 4:
                tempArr = valueBytes[alignment_skip : alignment_skip + 4]
                return round(struct.unpack("f", tempArr)[0], 0)
        return False

    @staticmethod
    def parseTwoInt(
        name: str, valueBytes: bytes, alignment_skip: int, aligned_size: int
    ) -> list[int] | Literal[False]:
        if persist.isA(persist.TYPE_TWOINT, name):
            if aligned_size == 8:
                tempArr = valueBytes[alignment_skip : alignment_skip + 8]
                return [
                    int.from_bytes(tempArr[0:4], "little"),
                    int.from_bytes(tempArr[4:4], "little"),
                ]
        return False

    @staticmethod
    def parse_hardcoded_type(
        name: str, valueBytes: bytes, alignment_skip: int, aligned_size: int
    ):
        return (
            persist.parseFloatArray(name, valueBytes, alignment_skip, aligned_size)
            or persist.parseIntVector(name, valueBytes, alignment_skip, aligned_size)
            or persist.parseStringVector(name, valueBytes, alignment_skip, aligned_size)
            or persist.parseFloat(name, valueBytes, alignment_skip, aligned_size)
            or persist.parseTwoInt(name, valueBytes, alignment_skip, aligned_size)
        )

    @staticmethod
    def persist_parser(file: str | Path | bytes):
        meta1nodeManager = Meta1nodeManager()
        current_node_id = 0
        if isinstance(file, bytes):
            fp = BytesIO(file)
        else:
            with open(file, "rb") as fb:
                raw_data = fb.read()
                fp = BytesIO(raw_data)
        fp.seek(8, 1)
        headerLength = int.from_bytes(fp.read(4), "little")
        if headerLength != 64:
            logger.error("Header Length is not 64: " + str(headerLength))
            return False
        fp.seek(8, 1)
        numMeta1Entries = int.from_bytes(fp.read(4), "little")
        meta1Offset = int.from_bytes(fp.read(4), "little")
        fp.seek(16, 1)
        numMeta2Entries = int.from_bytes(fp.read(4), "little")
        meta2Offset = int.from_bytes(fp.read(4), "little")
        fp.seek(8, 1)
        dataOffset = int.from_bytes(fp.read(4), "little")
        fp.seek(0, 2)
        dataLength = fp.tell()
        # print(headerLength, meta1Offset, numMeta2Entries, meta2Offset, dataOffset)

        fp.seek(meta1Offset, 0)
        meta1nodeManager.parse_raw(
            BytesIO(fp.read(numMeta1Entries * 16)), numMeta1Entries
        )
        logger.debug(meta1nodeManager.to_dict())

        fp.seek(meta2Offset, 0)
        meta2DataLength = dataOffset - meta2Offset
        if meta2DataLength % 12 != 0:
            logger.error("Meta2 has wrong number of bytes: " + str(meta2DataLength))
            return False
        meta2List: list[tuple[int, int, int]] = []
        for _ in range(numMeta2Entries):
            entryHash = int.from_bytes(fp.read(4), "little")
            entry_relative_offset = int.from_bytes(fp.read(4), "little")
            fieldInfo = int.from_bytes(fp.read(4), "little")
            meta2List.append((entryHash, entry_relative_offset, fieldInfo))

        for x in range(numMeta2Entries):
            meta2_entry = meta2List[x]
            entry_relative_offset = meta2_entry[1]
            is_raw = meta2_entry[2] & 0b1
            name_length = (meta2_entry[2] & 0b11111111100) >> 2
            meta1_index = (meta2_entry[2] & 0b1111111111111111111100000000000) >> 11

            entry_offset = dataOffset + entry_relative_offset
            alignment_skip = (4 - (entry_offset + name_length) % 4) % 4
            entry_relative_data_offset = entry_relative_offset + name_length
            next_relative_entry_offset = (
                meta2List[x + 1][1]
                if x + 1 < len(meta2List)
                else dataLength - dataOffset
            )
            value_length = next_relative_entry_offset - entry_relative_data_offset
            aligned_size = value_length - alignment_skip

            fp.seek(entry_offset, 0)
            nameBytes = fp.read(name_length - 1)
            fp.seek(1, 1)
            fp.seek(alignment_skip, 1)
            name = bytes.decode(nameBytes, "utf-8")
            # if name == "raw_data":
            #     logger.setLevel(logging.DEBUG)
            # else:
            #     logger.setLevel(logging.INFO)

            def get_value(
                is_raw: int,
                entry_offset: int,
                name: str,
                name_length: int,
                value_length: int,
                alignment_skip: int,
                aligned_size: int,
            ) -> (
                str
                | int
                | bytes
                | List[bool]
                | list[float]
                | list[int]
                | float
                | list[str]
                | Dict[str, Any]
            ):
                if is_raw:
                    return ""
                fp.seek(entry_offset + name_length, 0)
                valueBytes = fp.read(value_length)
                if name == "raw_data":
                    logger.debug(valueBytes[alignment_skip + 4 :])
                    return persist.persist_parser(valueBytes[alignment_skip + 4 :])
                if hardcoded_value := persist.parse_hardcoded_type(
                    name, valueBytes, alignment_skip, aligned_size
                ):
                    return hardcoded_value
                if value_length == 1:
                    logger.debug(valueBytes[0])
                    if valueBytes[0] >= 0x20 and valueBytes[0] <= 0x7E:
                        return bytes.decode(valueBytes, "utf-8")
                    else:
                        return valueBytes[0] != 0x00

                if (
                    aligned_size == 8
                    and (
                        valueBytes[alignment_skip + 0] == 0x00
                        or valueBytes[alignment_skip + 0] == 0x01
                    )
                    and (
                        valueBytes[alignment_skip + 4] == 0x00
                        or valueBytes[alignment_skip + 4] == 0x01
                    )
                ):
                    return [
                        valueBytes[alignment_skip + 0] != 0x00,
                        valueBytes[alignment_skip + 4] != 0x00,
                    ]

                if aligned_size == 4:
                    return int.from_bytes(
                        valueBytes[alignment_skip : alignment_skip + 4], "little"
                    )
                if aligned_size >= 5:
                    str_len = int.from_bytes(
                        valueBytes[alignment_skip : alignment_skip + 4], "little"
                    )
                    logger.debug(str_len)
                    if aligned_size == str_len + 4:
                        strBytes = valueBytes[
                            alignment_skip + 4 : alignment_skip + 4 + str_len - 1
                        ]
                        try:
                            return bytes.decode(strBytes, "utf-8")
                        except Exception:
                            pass
                try:
                    return bytes.decode(valueBytes, "utf-8")
                except Exception:
                    return valueBytes

            value = get_value(
                is_raw,
                entry_offset,
                name,
                name_length,
                value_length,
                alignment_skip,
                aligned_size,
            )

            logger.debug(
                f"is_raw:{is_raw}\tnameLength:{name_length}\tmeta1_block_index:{meta1_index}\talignment:{4 - (dataOffset + meta2_entry[1] + name_length) % 4}\tvalueLength{value_length}"
            )
            logger.debug(f"name:{name}\tvalue:{value}")
            if is_raw:
                if node := meta1nodeManager.get_node(meta1_index):
                    node.set_key(name)
                    current_node_id = meta1_index
            else:
                if node := meta1nodeManager.get_node(current_node_id):
                    node.add_value({name: value})

        return meta1nodeManager.to_dict()


class xml_data:
    mod_title: str
    mod_versions: List[int]
    mod_tags: List[str]
    mod_description: str
    mod_PublishedFileId: str

    def __init__(
        self,
        mod_title: str,
        mod_versions: List[int],
        mod_tags: List[str],
        mod_description: str,
        mod_PublishedFileId: str,
    ):
        self.mod_title = mod_title
        self.mod_versions = mod_versions
        self.mod_tags = mod_tags
        self.mod_description = mod_description
        self.mod_PublishedFileId = mod_PublishedFileId

    @classmethod
    def etree_text_iter(cls, tree: Element, name: str):
        for elem in tree.iter(name):
            if isinstance(elem.text, str):
                return elem.text
        return ""

    @classmethod
    def mod_xml_parser(cls, xml_file: str | Path):
        mod_title: str = ""
        mod_versions: List[int] = [0, 0, 0]
        mod_tags: List[str] = []
        mod_description: str = ""
        mod_PublishedFileId: str = ""
        try:
            tree = ET.fromstring(
                Path(xml_file).read_text(encoding="utf-8", errors="ignore").strip()
            )
        except Exception:
            return cls(
                mod_title, mod_versions, mod_tags, mod_description, mod_PublishedFileId
            )
        root = tree
        mod_title = cls.etree_text_iter(root, "Title") or mod_title
        mod_title = re.sub(r'[\/:*?"<>|]', "_", mod_title).strip()
        mod_versions[0] = int(
            float(cls.etree_text_iter(root, "VersionMajor") or mod_versions[0])
        )
        mod_versions[1] = int(
            float(cls.etree_text_iter(root, "VersionMinor") or mod_versions[1])
        )
        mod_versions[2] = int(
            float(cls.etree_text_iter(root, "TargetBuild") or mod_versions[2])
        )
        mod_description = (
            cls.etree_text_iter(root, "ItemDescription") or mod_description
        )
        mod_PublishedFileId = (
            cls.etree_text_iter(root, "PublishedFileId") or mod_PublishedFileId
        )
        for Tags in root.iter("Tags"):
            if not isinstance(Tags.text, str) or not Tags.text.strip():
                continue
            mod_tags.append(Tags.text)
        return cls(
            mod_title, mod_versions, mod_tags, mod_description, mod_PublishedFileId
        )


class DarkestDungeonModDataChecker(mobase.ModDataChecker):
    def __init__(self):
        super().__init__()
        self.validDirNames = [
            "activity_log",
            "audio",
            "campaign",
            "colours",
            "curios",
            "cursors",
            "dlc",
            "dungeons",
            "effects",
            "fe_flow",
            "fonts",
            "fx",
            "game_over",
            "heroes",
            "inventory",
            "loading_screen",
            "localization",
            "loot",
            "maps",
            "modes",
            "monsters",
            "overlays",
            "panels",
            "props",
            "raid",
            "raid_result",
            "scripts",
            "scrolls",
            "shaders",
            "shared",
            "trinkets",
            "upgrades",
            "video",
        ]
        self.invalidFileNames = ["project.xml", "preview_icon.png", "modfiles.txt"]

    def dataLooksValid(
        self, filetree: mobase.IFileTree
    ) -> mobase.ModDataChecker.CheckReturn:
        for entry in filetree:
            if entry.name().casefold() in self.invalidFileNames:
                return mobase.ModDataChecker.FIXABLE
        for entry in filetree:
            if not entry.isDir():
                continue
            if entry.name().casefold() in self.validDirNames:
                return mobase.ModDataChecker.VALID
        return mobase.ModDataChecker.INVALID

    def fix(self, filetree: mobase.IFileTree) -> mobase.IFileTree:
        id = random.randint(1, 99999)
        for entry in list(filetree):
            if entry.name().casefold() == "project.xml":
                filetree.move(entry, f"project_file/{id}.xml")
            if entry.name().casefold() == "preview_icon.png":
                filetree.move(entry, f"preview_file/{id}.png")
            if entry.name().casefold() == "modfiles.txt":
                entry.detach()
        return filetree


class DarkestDungeonModDataContent(mobase.ModDataContent):
    def __init__(self, modPath: str):
        super().__init__()
        self.modPath = modPath
        self.modIdMap: Dict[str, int] = {}

    def getAllContents(self) -> list[mobase.ModDataContent.Content]:
        contents: List[mobase.ModDataContent.Content] = []
        icons = [i for i in Path(self.modPath).glob("*/preview_file/*.png")]
        for icon, index in zip(icons, range(1, 1 + len(icons)), strict=True):
            pass
            self.modIdMap[icon.parent.parent.name] = index
            contents.append(mobase.ModDataContent.Content(index, "缩略图", str(icon)))
        icons = [i for i in Path(self.modPath).glob("*/preview_icon.png")]
        for icon, index in zip(icons, range(1, 1 + len(icons)), strict=True):
            pass
            self.modIdMap[icon.parent.name] = index
            contents.append(mobase.ModDataContent.Content(index, "缩略图", str(icon)))
        return contents

    def getContentsFor(self, filetree: mobase.IFileTree) -> list[int]:
        if filetree.name() in self.modIdMap:
            return [self.modIdMap[filetree.name()]]
        return []


class DarkestDungeonSaveGame(BasicGameSaveGame):
    def __init__(self, filepath: Path):
        super().__init__(filepath)
        self._filepath = filepath
        dataPath: Path = filepath.joinpath("persist.game.json")
        self.name: str = ""
        if self.isBinary(dataPath):
            self.loadBinarySaveFile(dataPath)
        else:
            self.loadJSONSaveFile(dataPath)

    def allFiles(self) -> list[str]:
        return [str(i) for i in self._filepath.rglob("*.json")]

    @staticmethod
    def isBinary(dataPath: Path) -> bool:
        with dataPath.open(mode="rb") as fp:
            magic: bytes = fp.read(4)
            # magic number in binary save files
            return magic == b"/x01/xb1/x00/x00" or magic == b"\x01\xb1\x00\x00"

    def loadJSONSaveFile(self, dataPath: Path):
        text = dataPath.read_text()
        content = json.loads(text)
        data = content["data"]
        self.name = str(data["estatename"])

    def loadBinarySaveFile(self, dataPath: Path):
        # see https://github.com/robojumper/DarkestDungeonSaveEditor
        with dataPath.open(mode="rb") as fp:
            # read Header

            # skip to headerLength
            fp.seek(8, 0)
            headerLength = int.from_bytes(fp.read(4), "little")
            if headerLength != 64:
                raise ValueError("Header Length is not 64: " + str(headerLength))
            fp.seek(4, 1)

            # meta1Size = int.from_bytes(fp.read(4), "little")
            fp.seek(4, 1)
            # numMeta1Entries = int.from_bytes(fp.read(4), "little")
            fp.seek(4, 1)

            meta1Offset = int.from_bytes(fp.read(4), "little")
            fp.seek(16, 1)
            numMeta2Entries = int.from_bytes(fp.read(4), "little")
            meta2Offset = int.from_bytes(fp.read(4), "little")
            fp.seek(4, 1)

            # dataLength = int.from_bytes(fp.read(4), "little")
            fp.seek(4, 1)

            dataOffset = int.from_bytes(fp.read(4), "little")

            # read Meta1 Block
            fp.seek(meta1Offset, 0)
            meta1DataLength = meta2Offset - meta1Offset
            if meta1DataLength % 16 != 0:
                raise ValueError(
                    "Meta1 has wrong number of bytes: " + str(meta1DataLength)
                )

            # read Meta2 Block
            fp.seek(meta2Offset, 0)
            meta2DataLength = dataOffset - meta2Offset
            if meta2DataLength % 12 != 0:
                raise ValueError(
                    "Meta2 has wrong number of bytes: " + str(meta2DataLength)
                )
            meta2List: list[tuple[int, int, int]] = []
            for _ in range(numMeta2Entries):
                entryHash = int.from_bytes(fp.read(4), "little")
                offset = int.from_bytes(fp.read(4), "little")
                fieldInfo = int.from_bytes(fp.read(4), "little")
                meta2List.append((entryHash, offset, fieldInfo))

            # read Data
            fp.seek(dataOffset, 0)
            for x in range(numMeta2Entries):
                meta2Entry = meta2List[x]
                fp.seek(dataOffset + meta2Entry[1], 0)
                nameLength = (meta2Entry[2] & 0b11111111100) >> 2
                # null terminated string
                nameBytes = fp.read(nameLength - 1)
                fp.seek(1, 1)
                name = bytes.decode(nameBytes, "utf-8")
                if name != "estatename":
                    continue
                valueLength = int.from_bytes(fp.read(4), "little")
                valueBytes = fp.read(valueLength - 1)
                value = bytes.decode(valueBytes, "utf-8")
                self.name = value
                break

    def getName(self) -> str:
        if self.name == "":
            return super().getName()
        return self.name


class DarkestDungeonLocalSavegames(mobase.LocalSavegames):
    def __init__(self):
        super().__init__()

    def mappings(self, profile_save_dir: QDir):
        source = profile_save_dir.absolutePath()
        destinations = [
            f"{QStandardPaths.standardLocations(QStandardPaths.StandardLocation.DocumentsLocation)[0]}\\Darkest",
        ]
        logger.debug(f"mapping {source} to {destinations[0]}")
        return [
            mobase.Mapping(
                source=source,
                destination=destination,
                is_directory=True,
                create_target=True,
            )
            for destination in destinations
        ]

    def prepareProfile(self, profile: mobase.IProfile) -> bool:
        return profile.localSavesEnabled()


class DarkestDungeonSaveGameInfoWidget(mobase.ISaveGameInfoWidget):
    GAME_MODE = {
        "base": "极暗",
        "radiant": "光耀",
        "new_game_plus": "狱火",
        "bloodmoon": "血月",
    }

    def __init__(self, parent: QWidget, game: BasicGame):
        super().__init__(parent)
        self._game_path = Path(game.gameDirectory().absolutePath())
        self._Vlayout = QVBoxLayout()
        self._save_name = QLabel()
        self._game_mode = QLabel()
        self._date_time = QLabel()
        self._isin_raid = QLabel()
        self._total_week = QLabel()
        self._estate = {
            i: QLabel()
            for i in [
                "gold",
                "bust",
                "portrait",
                "deed",
                "crest",
                "shard",
                "memory",
                "blueprint",
            ]
        }
        self._estate_pos = {
            "gold": self._game_path / "shared" / "estate" / "currency.gold.icon.png",
            "bust": self._game_path / "shared" / "estate" / "currency.bust.icon.png",
            "portrait": self._game_path
            / "shared"
            / "estate"
            / "currency.portrait.icon.png",
            "deed": self._game_path / "shared" / "estate" / "currency.deed.icon.png",
            "crest": self._game_path / "shared" / "estate" / "currency.crest.icon.png",
            "shard": self._game_path / "shared" / "estate" / "currency.shard.icon.png",
            "memory": self._game_path
            / "dlc"
            / "735730_color_of_madness"
            / "shared"
            / "estate"
            / "currency.memory.icon.png",
            "blueprint": self._game_path
            / "dlc"
            / "580100_crimson_court"
            / "features"
            / "districts"
            / "shared"
            / "estate"
            / "currency.blueprint.icon.png",
        }
        first_H = QHBoxLayout()
        second_H = QHBoxLayout()
        third_H = QHBoxLayout()
        fourth_H = QHBoxLayout()
        first_H.addWidget(QLabel("存档:"))
        first_H.addWidget(self._save_name)
        second_H.addWidget(QLabel("难度:"))
        second_H.addWidget(self._game_mode)
        second_H.addWidget(QLabel("当前在:"))
        second_H.addWidget(self._isin_raid)
        third_H.addWidget(QLabel("最后游玩时间:"))
        third_H.addWidget(self._date_time)
        for k, v in self._estate.items():
            if self._estate_pos[k].exists():
                label = QLabel()
                pixmap = QPixmap(str(self._estate_pos[k]))
                pixmap.scaledToWidth(10)
                label.setPixmap(pixmap)
                fourth_H.addWidget(label)
                fourth_H.addWidget(v)
            else:
                logger.error(f"{self._estate_pos[k]} no exist")
        first_H.addStretch()
        second_H.addStretch()
        third_H.addStretch()
        fourth_H.addStretch()
        self._Vlayout.addLayout(first_H)
        self._Vlayout.addLayout(second_H)
        self._Vlayout.addLayout(third_H)
        self._Vlayout.addLayout(fourth_H)
        self.setLayout(self._Vlayout)

    def setSave(self, save: mobase.ISaveGame):
        save_path = Path(save.getFilepath())
        game_data = persist.persist_parser(save_path / "persist.game.json")
        estate_data = persist.persist_parser(save_path / "persist.estate.json")
        if game_data and estate_data:
            self.hide()
            self._save_name.clear()
            self._game_mode.clear()
            self._date_time.clear()
            self._isin_raid.clear()
            self._save_name.setText(f"{game_data['base_root']['estatename']}")
            if game_data["base_root"]["game_mode"] in self.GAME_MODE:
                self._game_mode.setText(
                    f"{self.GAME_MODE[game_data['base_root']['game_mode']]}"
                )
            else:
                self._game_mode.setText(f"{game_data['base_root']['game_mode']}")
            self._date_time.setText(f"{game_data['base_root']['date_time']}")
            self._isin_raid.setText(
                "地牢" if game_data["base_root"]["inraid"] else "小镇"
            )
            for v in estate_data["base_root"]["wallet"].values():
                if v["type"] in self._estate:
                    self._estate[v["type"]].setText(f"{v['amount']:,}")
            self.setWindowFlags(
                Qt.WindowType.ToolTip | Qt.WindowType.BypassGraphicsProxyWidget
            )
            self.show()
        else:
            self.hide()


class DarkestDungeonSaveGameInfo(mobase.SaveGameInfo):
    def __init__(self, game: BasicGame):
        super().__init__()
        self._game = game

    def getMissingAssets(
        self: "DarkestDungeonSaveGameInfo", save: mobase.ISaveGame
    ) -> Dict[str, Sequence[str]]:
        return {}

    def getSaveGameWidget(
        self: "DarkestDungeonSaveGameInfo", parent: QWidget
    ) -> DarkestDungeonSaveGameInfoWidget | None:
        return DarkestDungeonSaveGameInfoWidget(parent, self._game)


class DarkestDungeonGame(BasicGame, mobase.IPluginFileMapper):
    Name = "DarkestDungeon"
    Author = "LIC"
    Version = "1.0.0"

    GameName = "Darkest Dungeon"
    GameShortName = "darkestdungeon"
    GameNexusName = "darkestdungeon"
    GameNexusId = 804
    GameSteamId = 262060
    GameGogId = 1719198803
    GameBinary = "_windowsnosteam//darkest.exe"
    GameDataPath = r"mods\!!MOD"
    # GameDataPath = r""
    GameSupportURL = (
        r"https://github.com/ModOrganizer2/modorganizer-basic_games/wiki/"
        "Game:-Darkest-Dungeon"
    )

    def __init__(self):
        BasicGame.__init__(self)
        mobase.IPluginFileMapper.__init__(self)
        self._organizer: mobase.IOrganizer = None  # type: ignore
        self.merge_to_one_json = [
            regex_json_data(
                "trinkets/*rarities.trinkets.json",
                ["id"],
                "trinkets/0000.rarities.trinkets.json",
            ),
            regex_json_data(
                "trinkets/*entries.trinkets.json",
                ["id"],
                "trinkets/0000.entries.trinkets.json",
            ),
            regex_json_data(
                "raid/ai/*monster_brains.json",
                ["id"],
                "raid/ai/0000.monster_brains.json",
            ),
            regex_json_data(
                "shared/buffs/*buffs.json",
                ["id"],
                "shared/buffs/0000.buffs.json",
            ),
        ]
        self.merge_to_one_json_necessary = self.merge_to_one_json[0:0]
        self.merge_same_json = [
            regex_json_data(
                "trinkets/*rarities.trinkets.json",
                ["id"],
                "",
            ),
            regex_json_data(
                "trinkets/*entries.trinkets.json",
                ["id"],
                "",
            ),
            regex_json_data("scripts/*raid_settings.json", ["key"], ""),
            regex_json_data(
                "raid/ai/*monster_brains.json",
                ["id"],
                "",
            ),
            regex_json_data(
                "campaign/quest/*quest.plot_quests.json",
                ["id"],
                "",
            ),
            regex_json_data(
                "loot/*loot.json",
                ["id", "difficulty", "dungeon"],
                "",
            ),
            regex_json_data(
                "shared/quirk/*quirk_library.json",
                ["id"],
                "",
            ),
            regex_json_data(
                "shared/buffs/*buffs.json",
                ["id"],
                "",
            ),
            regex_json_data(
                "shared/quirk/*quirk_act_outs.json",
                ["quirk_id"],
                "",
            ),
        ]
        self.merge_same_json_necessary = self.merge_same_json

    def local_saves_directory(self) -> List[Path]:
        self._organizer.profilePath()
        docSaves = Path.home() / "Saved Games" / "Darkest"
        if (steamDir := find_steam_path()) is None:
            logger.info("Steam not found")
            return [docSaves]
        for child in steamDir.joinpath("userdata").iterdir():
            if not child.is_dir() or child.name == "0":
                continue
            steamSaves = child.joinpath("262060")
            if steamSaves.is_dir():
                return [docSaves, steamSaves]
        logger.info("Steam saves not found")
        return [docSaves]

    def init(self, organizer: mobase.IOrganizer) -> bool:
        super().init(organizer)
        self._organizer = organizer
        self._register_feature(DarkestDungeonModDataChecker())
        self._register_feature(DarkestDungeonModDataContent(organizer.modsPath()))
        self._register_feature(DarkestDungeonSaveGameInfo(self))
        self._register_feature(DarkestDungeonLocalSavegames())
        organizer.pluginList().onRefreshed(self.Refreshed)
        organizer.onAboutToRun(self.shutdown_when_steam_not_running)
        organizer.onFinishedRun(self.clean_empty_json)

        (Path(self._organizer.basePath()) / "logs").mkdir(parents=True, exist_ok=True)
        log_handler = logging.FileHandler(
            Path(self._organizer.basePath()) / "logs" / "darkestdungeon.log",
            mode="w+",
            encoding="utf-8",
        )
        log_handler.setFormatter(
            logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
        )
        log_handler.setLevel(logging.DEBUG)
        if len(logger.handlers) <= 1:
            logger.addHandler(log_handler)
        return True

    def clean_empty_json(self, exe_path: str, exit_code: int):
        for source in self.merge_to_one_json:
            for file in self._get_overwrite_path().glob(source.regex):
                if (
                    file.absolute()
                    != (self._get_overwrite_path() / source.file_name).absolute()
                ) and file.stat().st_size == 0:
                    file.unlink()
                    logger.debug(f"unlink {file}")

    def is_steam_runing(self) -> bool:
        for pid in psutil.pids():
            try:
                if psutil.Process(pid).name() == "steam.exe":
                    return True
            except Exception:
                pass
        return False

    def shutdown_when_steam_not_running(
        self, file: str, location: QDir, arguments: str
    ) -> bool:
        if not self.is_steam_runing() and Path(file).name == "darkest.exe":
            raise OSError(
                "Steam 没有运行！ 先运行Steam！\nSteam is not running! RUN STEAM FIRST!"
            )
            return False
        return True

    def _get_overwrite_path(self):
        return Path(self._organizer.overwritePath())

    def _get_game_path(self):
        return Path(self.gameDirectory().absolutePath())

    def _get_workshop_path(self):
        workshop_paths: list[Path] = []
        steam_path = find_steam_path()
        if steam_path is not None:
            library_folders = parse_library_info(
                steam_path / "steamapps" / "libraryfolders.vdf"
            )
            for library_folder in library_folders:
                acf_file = (
                    library_folder.path
                    / "steamapps"
                    / "workshop"
                    / "appworkshop_262060.acf"
                )
                if acf_file.exists():
                    workshop_paths.append(
                        library_folder.path / "steamapps" / "workshop"
                    )
        else:
            workshop_paths.append(find_games()["262060"].parent.parent / "workshop")
        logger.debug(f"Found {len(workshop_paths)} workshop: {workshop_paths}")
        return workshop_paths

    def _get_mo_mods_path(self):
        return Path(self._organizer.modsPath())

    def Refreshed(self):
        logger.info("refreshing")
        workshop_path_workshop_items: Dict[Path, Dict[str, Dict[str, str]]] = {}
        for workshop_path in self._get_workshop_path():
            acf_path = workshop_path / "appworkshop_262060.acf"
            if acf_path.exists():
                workshop_path_workshop_items[workshop_path] = vdf.load(open(acf_path))[  # type: ignore
                    "AppWorkshop"
                ]["WorkshopItemDetails"]
                logger.debug(
                    f"found {len(workshop_path_workshop_items[workshop_path])} mod-records in {workshop_path}"
                )
            else:
                logger.debug(f"darkest_dungeon acf file not exist in {workshop_path}")
        mod_list = self._organizer.modList()
        mod_names = mod_list.allMods()
        mo_mod_path = Path(self._organizer.modsPath())
        mo_workshop_PublishedFileId: Dict[str, mobase.IModInterface] = {
            str(i.stem.strip("w")): mod_list.getMod(str(i.parent.parent.name))
            for i in mo_mod_path.glob("*/project_file/w*.manifest")
        }
        logger.debug(f"found {len(mo_workshop_PublishedFileId)} workshop mods in mo2")
        mo_local_PublishedFileId: Dict[str, mobase.IModInterface] = {
            str(i.stem.strip("l")): mod_list.getMod(str(i.parent.parent.name))
            for i in mo_mod_path.glob("*/project_file/l*.manifest")
        }

        if not (self._get_game_path() / Path(self.GameDataPath)).exists():
            (self._get_game_path() / Path(self.GameDataPath)).mkdir(
                parents=True, exist_ok=True
            )

        for mod_name in mod_names:
            mod = mod_list.getMod(mod_name)
            mod_folder = mo_mod_path / mod_name
            preview_file = mod_folder / "preview_icon.png"
            xml_file = [i for i in (mod_folder / "project_file").glob("*.xml")] + [
                i for i in mod_folder.glob("project.xml")
            ]
            xml_file = xml_file[0] if xml_file else Path("None")
            manifest_file = [
                i for i in (mod_folder / "project_file").glob("*.manifest")
            ]
            manifest_file = manifest_file[0] if manifest_file else Path("None")

            # txt_file = mod_folder / "modfiles.txt"
            # log_file = mod_folder / "steam_workshop_uploader.log"
            # # remove steam_workshop_uploader.log
            # if log_file.exists():
            #     log_file.unlink()
            # # remove modfiles.txt
            # if txt_file.exists():
            #     txt_file.unlink()

            if not xml_file.exists():
                continue

            mod_xml_data = xml_data.mod_xml_parser(xml_file)

            # set version
            if not mod.version().isValid():
                mod.setVersion(mobase.VersionInfo(*mod_xml_data.mod_versions))

            # add categories
            if not mod.categories():
                for i in mod_xml_data.mod_tags:
                    if i not in mod.categories():
                        mod.addCategory(str(i))
                if (
                    manifest_file.exists()
                    and manifest_file.name.startswith("w")
                    and "workshop mod" not in mod.categories()
                ):
                    mod.addCategory("workshop mod")
                if (
                    manifest_file.exists()
                    and manifest_file.name.startswith("l")
                    and "local mod" not in mod.categories()
                ):
                    mod.addCategory("local mod")

            # set url
            if (
                not mod.url()
                and manifest_file.exists()
                and manifest_file.name.startswith("w")
            ):
                mod.setUrl(
                    f"https://steamcommunity.com/sharedfiles/filedetails/?id={mod_xml_data.mod_PublishedFileId}"
                )

        # copy steam workshop mods
        # 使用mod_PublishedFileId保证唯一，可能会出现本地mod占用的问题
        # game_workshop_path = self._get_workshop_path() / "content" / "262060"
        for game_workshop_path, workshop_items in workshop_path_workshop_items.items():
            for PublishedFileId in set(workshop_items.keys()) - set(
                mo_workshop_PublishedFileId
            ):
                xml_file = (
                    game_workshop_path
                    / "content"
                    / "262060"
                    / PublishedFileId
                    / "project.xml"
                )
                if not xml_file.exists():
                    logger.debug(
                        f"worshop mod Id {PublishedFileId} not found in {game_workshop_path}"
                    )
                    continue
                mod_title = xml_data.mod_xml_parser(xml_file).mod_title
                logger.info(f"Adding workshop mod {mod_title}")
                if mod_title := util.scopy_mod(
                    game_workshop_path / "content" / "262060" / PublishedFileId,
                    mo_mod_path / mod_title,
                ):
                    mo_mod_folder = mo_mod_path / mod_title
                    log_file = mo_mod_folder / "steam_workshop_uploader.log"
                    txt_file = mo_mod_folder / "modfiles.txt"
                    xml_file = mo_mod_folder / "project.xml"
                    preview_file = mo_mod_folder / "preview_icon.png"
                    manifest_file = (
                        mo_mod_folder / "project_file" / f"w{PublishedFileId}.manifest"
                    )

                    (mo_mod_folder / "preview_file").mkdir(exist_ok=True)
                    (mo_mod_folder / "project_file").mkdir(exist_ok=True)

                    if txt_file.exists():
                        txt_file.unlink()
                    if log_file.exists():
                        log_file.unlink()

                    if preview_file.exists():
                        preview_file.rename(
                            mo_mod_folder / "preview_file" / f"{PublishedFileId}.png"
                        )

                    if xml_file.exists():
                        xml_file.rename(
                            mo_mod_folder / "project_file" / f"{PublishedFileId}.xml"
                        )
                        manifest_file.write_text(
                            workshop_items[PublishedFileId]["manifest"]
                        )

                else:
                    logger.info(f"Failed to add mod {mod_title}")
                logger.info(f"Added mod {mod_title}")

        # check upgrade
        for game_workshop_path, workshop_items in workshop_path_workshop_items.items():
            for PublishedFileId in set([i for i in workshop_items.keys()]) & set(
                [i for i in mo_workshop_PublishedFileId]
            ):
                mod = mo_workshop_PublishedFileId[PublishedFileId]
                if not mod:
                    continue

                old_manifest_file = (
                    Path(mod.absolutePath())
                    / "project_file"
                    / f"w{PublishedFileId}.manifest"
                )

                new_manifest = workshop_items[PublishedFileId]["manifest"]
                if old_manifest_file.exists():
                    old_manifest = old_manifest_file.read_text()
                else:
                    old_manifest = ""

                if new_manifest != old_manifest:
                    xml_file = (
                        game_workshop_path
                        / "content"
                        / "262060"
                        / PublishedFileId
                        / "project.xml"
                    )
                    if xml_file.exists():
                        mod.setNewestVersion(
                            mobase.VersionInfo(
                                *xml_data.mod_xml_parser(
                                    game_workshop_path
                                    / "content"
                                    / "262060"
                                    / PublishedFileId
                                    / "project.xml"
                                ).mod_versions
                            )
                        )

        # copy local mods
        for xml_file in (Path(self.gameDirectory().absolutePath()) / "mods").glob(
            "*/project.xml"
        ):
            mod_xml_data = xml_data.mod_xml_parser(xml_file)
            mod_PublishedFileIds = [
                i.stem.strip("l") for i in xml_file.parent.glob("*.manifest")
            ]
            mo_mod_folder = mo_mod_path / mod_xml_data.mod_title
            id = str(random.randint(1, 9999999))
            if not mod_PublishedFileIds or not set(mod_PublishedFileIds) & set(
                mo_local_PublishedFileId.keys()
            ):
                logger.info(f"Adding mod {xml_file.parent.name}")
                mod_title = util.scopy_mod(xml_file.parent, mo_mod_folder)
                mo_mod_folder = mo_mod_path / mod_title
                (xml_file.parent / f"l{id}.manifest").write_text("", encoding="utf-8")
                logger.info(f"Added mod {xml_file.parent.name}")
            else:
                continue

            preview_file = mo_mod_folder / "preview_icon.png"
            txt_file = mo_mod_folder / "modfiles.txt"
            xml_file = mo_mod_folder / "project.xml"
            log_file = mo_mod_folder / "steam_workshop_uploader.log"

            if log_file.exists():
                log_file.unlink()
            if txt_file.exists():
                txt_file.unlink()

            (mo_mod_folder / "preview_file").mkdir(exist_ok=True)
            if preview_file.exists():
                preview_file.rename(mo_mod_folder / "preview_file" / f"{id}.png")

            (mo_mod_folder / "project_file").mkdir(exist_ok=True)
            if xml_file.exists():
                xml_file.rename(xml_file.parent / "project_file" / f"{id}.xml")
                open(
                    xml_file.parent / "project_file" / f"l{id}.manifest",
                    "w+",
                    encoding="utf-8",
                ).write("")

    def mappings(self) -> List[mobase.Mapping]:
        mod_titles = [
            i
            for i in self._organizer.modList().allModsByProfilePriority()
            if self._organizer.modList().state(i) & mobase.ModState.ACTIVE
        ]

        def create_project_xml():  # merge mod xml
            project_text = """
    <project>
        <PreviewIconFile>preview_icon.png</PreviewIconFile>
        <ItemDescriptionShort/>
        <ModDataPath></ModDataPath>
        <Title>merge mod</Title>
        <Language>english</Language>
        <UpdateDetails/>
        <Visibility>private</Visibility>
        <UploadMode>dont_submit</UploadMode>
        <VersionMajor>1</VersionMajor>
        <VersionMinor>0</VersionMinor>
        <TargetBuild>0</TargetBuild>
        <Tags/>
        <ItemDescription/>
        <PublishedFileId>0000000000</PublishedFileId>
    </project>"""
            xml_path = self._get_overwrite_path() / "project.xml"
            xml_path.write_text(project_text)

        def merge_effect_files():  # merge effect files
            effect_files: Dict[str, List[Path]] = defaultdict(list)
            overwrite_effect_folder = self._get_overwrite_path() / "effects"

            if not overwrite_effect_folder.exists():
                overwrite_effect_folder.mkdir()
            else:
                for file in overwrite_effect_folder.glob("*.effects.darkest"):
                    file.unlink()

            for mod_title in mod_titles:
                for effect_file in (
                    self._get_mo_mods_path() / mod_title / "effects"
                ).glob("*.effects.darkest"):
                    effect_files[effect_file.name].append(effect_file)

                    # region fix riposte_validate problem
                    trigger = False
                    darkest_data = darkest.paser(
                        effect_file.read_text(encoding="utf-8", errors="ignore")
                    )
                    for index, effect in enumerate(darkest_data.data["effect"]):
                        if (
                            "target" in effect
                            and (
                                effect["target"] == '"target"'
                                or effect["target"] == "'target'"
                            )
                            and "riposte" in effect
                            and effect["riposte"] == "1"
                            and "riposte_validate" not in effect
                        ):
                            darkest_data.data["effect"][index]["riposte_validate"] = (
                                "false"
                            )
                            trigger = True
                    if trigger:
                        open(
                            overwrite_effect_folder / f"{effect_file.name}",
                            "w+",
                            encoding="utf-8",
                        ).write(str(darkest_data))
                    # endregion

            for effect_file_name, files in effect_files.items():
                if len(files) > 1:
                    contents = "\n".join([util.try_read_text(i) for i in files])
                    open(
                        overwrite_effect_folder / f"{effect_file_name}",
                        "w+",
                        encoding="utf-8",
                    ).write(contents)

        def preload_static_resource():  # region mapping static resource files
            static_resource_mapping: List[mobase.Mapping] = []
            static_resource_folder = [
                "fe_flow",
                "fonts",
                "cursors",
                "overlays",
            ]
            for mod_title in mod_titles:
                for path in set(static_resource_folder) & set(
                    [i.name for i in (self._get_mo_mods_path() / mod_title).glob("*")]
                ):
                    static_resource_mapping.append(
                        mobase.Mapping(
                            str(self._get_mo_mods_path() / mod_title / path),
                            str(self._get_game_path() / path),
                            True,
                            True,
                        )
                    )
            return static_resource_mapping

        def preload_dynamic_resource() -> (
            List[mobase.Mapping]
        ):  # region mapping static resource files
            dynamic_resource_mapping: List[mobase.Mapping] = []
            dynamic_resource_folder_suffix = {
                "localization": ["loc2"],
            }
            for index, mod_title in enumerate(mod_titles):
                for folder, suffixs in dynamic_resource_folder_suffix.items():
                    for suffix in suffixs:
                        for file in (
                            self._get_mo_mods_path() / mod_title / folder
                        ).rglob(f"*.{suffix}"):
                            relative_path = file.parent.relative_to(
                                self._get_mo_mods_path() / mod_title
                            )
                            if file.stem.startswith(tuple([str(i) for i in range(9)])):
                                mapping_file_name = (
                                    f"999{index:03d}{file.stem}.{suffix}"
                                )
                            else:
                                mapping_file_name = file.name
                            # dynamic_resource_mapping.append(
                            #     mobase.Mapping(
                            #         str(file.absolute()),
                            #         str(
                            #             self._get_game_path()
                            #             / relative_path
                            #             / mapping_file_name
                            #         ),
                            #         True,
                            #         True,
                            #     )
                            # )
                            dynamic_resource_mapping.append(
                                mobase.Mapping(
                                    str(file.absolute()),
                                    str(
                                        self._get_game_path()
                                        / Path(self.GameDataPath)
                                        / relative_path
                                        / mapping_file_name
                                    ),
                                    True,
                                    True,
                                )
                            )
            return dynamic_resource_mapping

        def merge_same_json_file():
            for source in self.merge_same_json:
                for file in self._get_overwrite_path().glob(source.regex):
                    file.unlink()
            for source in self.merge_same_json_necessary:
                relative_path_file: dict[Path, list[Path]] = defaultdict(list)
                for mod_title in mod_titles:
                    for file in (self._get_mo_mods_path() / mod_title).glob(
                        source.regex
                    ):
                        relative_path_file[
                            file.relative_to(self._get_mo_mods_path() / mod_title)
                        ].append(file)
                for relative_path, files in relative_path_file.items():
                    if len(files) > 1:
                        logger.debug(f"merge {relative_path}")
                        result = util.smerge_jsons(files, source.identifier)
                        if not result:
                            logger.error(f"Failed to merge {relative_path}")
                            continue
                        overwrite_file = self._get_overwrite_path() / relative_path
                        overwrite_file.parent.mkdir(parents=True, exist_ok=True)
                        open(overwrite_file, "w+", encoding="utf-8").write(
                            json.dumps(result, ensure_ascii=False)
                        )

        def merge_regex_json_file():
            for source in self.merge_to_one_json:
                for file in self._get_overwrite_path().glob(source.regex):
                    file.unlink()
            for source in self.merge_to_one_json_necessary:
                logger.debug(f"merging regex {source.regex}")
                all_regex_files: list[Path] = []
                relative_paths: list[Path] = []
                for mod_title in mod_titles:
                    regex_files = list(
                        (self._get_mo_mods_path() / mod_title).glob(source.regex)
                    )
                    all_regex_files += regex_files
                    for file in regex_files:
                        relative_paths.append(
                            file.relative_to(self._get_mo_mods_path() / mod_title)
                        )
                        logger.debug(f"merging {file}")
                result = util.smerge_jsons(
                    all_regex_files,
                    source.identifier,
                )
                if not result:
                    logger.error(f"Failed to merge {source.file_name}")
                    continue
                overwrite_file = self._get_overwrite_path() / source.file_name
                overwrite_file.parent.mkdir(parents=True, exist_ok=True)
                for relative_path in relative_paths:
                    file_path = self._get_overwrite_path() / relative_path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.touch()
                    logger.debug(f"touch {file_path}")
                open(overwrite_file, "w+", encoding="utf-8").write(
                    json.dumps(result, ensure_ascii=False)
                )
                logger.debug(f"merge regex json file {overwrite_file} Done")

        logger.debug("create_project_xml start...")
        create_project_xml()
        logger.debug("create_project_xml end")
        logger.debug("merge_regex_json_file start...")
        merge_regex_json_file()
        logger.debug("merge_regex_json_file end")
        logger.debug("merge_same_json_file start...")
        merge_same_json_file()
        logger.debug("merge_same_json_file end")
        merge_effect_files()
        return preload_static_resource() + preload_dynamic_resource()

    def executables(self):
        if self.is_steam():
            path = QFileInfo(self.gameDirectory(), "_windows/darkest.exe")
        else:
            path = QFileInfo(self.gameDirectory(), "_windowsnosteam/darkest.exe")
        return [
            mobase.ExecutableInfo("Darkest Dungeon", path).withWorkingDirectory(
                self.gameDirectory()
            ),
        ]

    @staticmethod
    def getCloudSaveDirectory() -> str | None:
        steamPath = find_steam_path()
        if steamPath is None:
            return None

        userData = steamPath.joinpath("userdata")
        for child in userData.iterdir():
            name = child.name
            try:
                userID = int(name)
            except ValueError:
                userID = -1
            if userID == -1:
                continue
            cloudSaves = child.joinpath("262060", "remote")
            if cloudSaves.exists() and cloudSaves.is_dir():
                return str(cloudSaves)
        return None

    def savesDirectory(self) -> QDir:
        documentsSaves = QDir(
            "{}/Darkest".format(
                QStandardPaths.writableLocation(
                    QStandardPaths.StandardLocation.DocumentsLocation
                )
            )
        )
        if self.is_steam():
            cloudSaves = self.getCloudSaveDirectory()
            if cloudSaves is None:
                return documentsSaves
            return QDir(cloudSaves)
        return documentsSaves

    def listSaves(self, folder: QDir) -> list[mobase.ISaveGame]:
        folder = self.savesDirectory()
        profiles: list[Path] = list(Path(folder.absolutePath()).glob("profile_[0-8]"))
        # profile_9 是对战模式存档
        return [DarkestDungeonSaveGame(path) for path in profiles]
