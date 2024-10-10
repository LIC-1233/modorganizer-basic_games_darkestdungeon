import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element

import mobase
from PyQt6.QtCore import QDir, QFileInfo, QStandardPaths, qInfo

from ..basic_game import BasicGame, BasicGameSaveGame
from ..steam_utils import find_games, find_steam_path, parse_library_info


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
        tree = ET.parse(xml_file)
        if not tree:
            return cls(mod_title, mod_versions, mod_tags, mod_description, mod_PublishedFileId)
        root = tree.getroot()
        mod_title = cls.etree_text_iter(root, "Title") or mod_title
        mod_title = re.sub(r'[\/:*?"<>|]', "_", mod_title).strip()
        mod_versions[0] = int(cls.etree_text_iter(root, "VersionMajor") or mod_versions[0])
        mod_versions[1] = int(cls.etree_text_iter(root, "VersionMinor") or mod_versions[1])
        mod_versions[2] = int(cls.etree_text_iter(root, "TargetBuild") or mod_versions[2])
        mod_description = cls.etree_text_iter(root, "ItemDescription") or mod_description
        mod_PublishedFileId = cls.etree_text_iter(root, "PublishedFileId") or mod_PublishedFileId
        for Tags in root.iter("Tags"):
            if not isinstance(Tags.text, str) or not Tags.text.strip():
                continue
            mod_tags.append(Tags.text)
        return cls(mod_title, mod_versions, mod_tags, mod_description, mod_PublishedFileId)


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

    def dataLooksValid(self, filetree: mobase.IFileTree) -> mobase.ModDataChecker.CheckReturn:
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
        dataPath = filepath.joinpath("persist.game.json")
        self.name: str = ""
        if self.isBinary(dataPath):
            self.loadBinarySaveFile(dataPath)
        else:
            self.loadJSONSaveFile(dataPath)

    @staticmethod
    def isBinary(dataPath: Path) -> bool:
        with dataPath.open(mode="rb") as fp:
            magic = fp.read(4)
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
                raise ValueError("Meta1 has wrong number of bytes: " + str(meta1DataLength))

            # read Meta2 Block
            fp.seek(meta2Offset, 0)
            meta2DataLength = dataOffset - meta2Offset
            if meta2DataLength % 12 != 0:
                raise ValueError("Meta2 has wrong number of bytes: " + str(meta2DataLength))
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
    GameSupportURL = r"https://github.com/ModOrganizer2/modorganizer-basic_games/wiki/" "Game:-Darkest-Dungeon"

    def __init__(self):
        BasicGame.__init__(self)
        mobase.IPluginFileMapper.__init__(self)
        self._organizer: mobase.IOrganizer = None  # type: ignore

    def init(self, organizer: mobase.IOrganizer) -> bool:
        super().init(organizer)
        self._organizer = organizer
        self._register_feature(DarkestDungeonModDataChecker())
        self._register_feature(DarkestDungeonModDataContent(organizer.modsPath()))
        organizer.pluginList().onRefreshed(self.Refreshed)
        return True

    @staticmethod
    def try_read_text(file_path: Path) -> str:
        encodings_to_try = ["gbk", "utf-8", "iso-8859-1"]
        for encoding in encodings_to_try:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to decode {file_path} with known encodings.")

    def _get_overwrite_path(self):
        return Path(self._organizer.overwritePath())

    def _get_game_path(self):
        return Path(self.gameDirectory().absolutePath())

    def _get_game_workshop_path(self):
        steam_path = find_steam_path()
        if steam_path is not None:
            library_folders = parse_library_info(steam_path / "steamapps" / "libraryfolders.vdf")
            for library_folder in library_folders:
                acf_file = library_folder.path / "steamapps" / "workshop" / "appworkshop_262060.acf"
                if acf_file.exists():
                    return library_folder.path / "steamapps" / "workshop"
        return find_games()["262060"].parent.parent / "workshop"

    def _get_mo_mods_path(self):
        return Path(self._organizer.modsPath())

    def Refreshed(self):
        def acf_parser(
            acf_file: str | Path,
        ) -> Dict[str, Dict[str, Dict[str, Dict[str, str]]]]:
            try:
                acf_content = open(acf_file).read()
                acf_content = re.sub(r'(".*?")\t*', r"\g<1>:", acf_content)
                acf_content = "{" + acf_content + "}"
                acf_content = re.sub(r':(\n\t*")', r",\g<1>", acf_content)
                acf_content = re.sub(r":(\n\t*\})", r",\g<1>", acf_content)
                acf_content = re.sub(r"\}", r"},", acf_content)
                acf_content = re.sub(r",(\n\t*\})", r"\g<1>", acf_content)
                acf_content = acf_content.strip(",")
                return json.loads(acf_content)
            except Exception as e:
                raise ValueError(f"failed to parse acf {acf_file}") from e

        def scopy_mod(scr: str | Path, dst: str | Path) -> str:  # type: ignore
            scr = str(scr)
            dst = str(dst)
            if not Path(scr).exists():
                return ""
            if Path(dst).exists():
                return scopy_mod(scr, f"{dst}_copy")  # type: ignore
            else:
                shutil.copytree(scr, dst)
                return dst

        qInfo("refreshing")
        acf_path = self._get_game_workshop_path() / "appworkshop_262060.acf"
        workshop_items = acf_parser(acf_path)["AppWorkshop"]["WorkshopItemDetails"]
        mod_list = self._organizer.modList()
        mod_names = mod_list.allMods()
        mo_mod_path = Path(self._organizer.modsPath())
        mo_workshop_PublishedFileId: Dict[str, mobase.IModInterface] = {
            str(i.stem.strip("w")): mod_list.getMod(str(i.parent.parent.name)) for i in mo_mod_path.glob("*/project_file/w*.manifest")
        }
        mo_local_PublishedFileId: Dict[str, mobase.IModInterface] = {
            str(i.stem.strip("l")): mod_list.getMod(str(i.parent.parent.name)) for i in mo_mod_path.glob("*/project_file/l*.manifest")
        }

        if not (self._get_game_path() / "mods/!!MOD").exists():
            (self._get_game_path() / "mods/!!MOD").mkdir(parents=True, exist_ok=True)

        for mod_name in mod_names:
            mod = mod_list.getMod(mod_name)
            mod_folder = mo_mod_path / mod_name
            preview_file = mod_folder / "preview_icon.png"
            xml_file = [i for i in (mod_folder / "project_file").glob("*.xml")] + [i for i in mod_folder.glob("project.xml")]
            xml_file = xml_file[0] if xml_file else Path("None")
            manifest_file = [i for i in (mod_folder / "project_file").glob("*.manifest")]
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
                if manifest_file.exists() and manifest_file.name.startswith("w") and "workshop mod" not in mod.categories():
                    mod.addCategory("workshop mod")
                if manifest_file.exists() and manifest_file.name.startswith("l") and "local mod" not in mod.categories():
                    mod.addCategory("local mod")

            # set url
            if not mod.url() and manifest_file.exists() and manifest_file.name.startswith("w"):
                mod.setUrl(f"https://steamcommunity.com/sharedfiles/filedetails/?id={mod_xml_data.mod_PublishedFileId}")

        # copy steam workshop mods
        # 使用mod_PublishedFileId保证唯一，可能会出现本地mod占用的问题
        game_workshop_path = self._get_game_workshop_path() / "content" / "262060"
        for PublishedFileId in set([i for i in workshop_items.keys()]) - set([i for i in mo_workshop_PublishedFileId.keys()]):
            xml_file = game_workshop_path / PublishedFileId / "project.xml"
            if not xml_file.exists():
                continue
            mod_title = xml_data.mod_xml_parser(xml_file).mod_title
            qInfo(f"Adding workshop mod {ascii(mod_title)}")
            if mod_title := scopy_mod(game_workshop_path / PublishedFileId, mo_mod_path / mod_title):
                mo_mod_folder = mo_mod_path / mod_title
                log_file = mo_mod_folder / "steam_workshop_uploader.log"
                txt_file = mo_mod_folder / "modfiles.txt"
                xml_file = mo_mod_folder / "project.xml"
                preview_file = mo_mod_folder / "preview_icon.png"
                manifest_file = mo_mod_folder / "project_file" / f"w{PublishedFileId}.manifest"

                (mo_mod_folder / "preview_file").mkdir(exist_ok=True)
                (mo_mod_folder / "project_file").mkdir(exist_ok=True)

                if txt_file.exists():
                    txt_file.unlink()
                if log_file.exists():
                    log_file.unlink()

                if preview_file.exists():
                    preview_file.rename(mo_mod_folder / "preview_file" / f"{PublishedFileId}.png")

                if xml_file.exists():
                    xml_file.rename(mo_mod_folder / "project_file" / f"{PublishedFileId}.xml")
                    manifest_file.write_text(workshop_items[PublishedFileId]["manifest"])

            else:
                qInfo(f"Failed to add mod {ascii(mod_title)}")
            qInfo(f"Added mod {ascii(mod_title)}")

        # check upgrade
        for PublishedFileId in set([i for i in workshop_items.keys()]) & set([i for i in mo_workshop_PublishedFileId]):
            mod = mo_workshop_PublishedFileId[PublishedFileId]
            if not mod:
                continue

            old_manifest_file = Path(mod.absolutePath()) / "project_file" / f"w{PublishedFileId}.manifest"

            new_manifest = workshop_items[PublishedFileId]["manifest"]
            if old_manifest_file.exists():
                old_manifest = old_manifest_file.read_text()
            else:
                old_manifest = ""

            if new_manifest != old_manifest:
                mod.setNewestVersion(mobase.VersionInfo(*xml_data.mod_xml_parser(game_workshop_path / PublishedFileId / "project.xml").mod_versions))

        # copy local mods
        for xml_file in (Path(self.gameDirectory().absolutePath()) / "mods").glob("*/project.xml"):
            mod_xml_data = xml_data.mod_xml_parser(xml_file)
            mod_PublishedFileIds = [i.stem.strip("l") for i in xml_file.parent.glob("*.manifest")]
            mo_mod_folder = mo_mod_path / mod_xml_data.mod_title
            id = str(random.randint(1, 9999999))
            if not mod_PublishedFileIds or not set(mod_PublishedFileIds) & set(mo_local_PublishedFileId.keys()):
                qInfo(f"Adding mod {ascii(xml_file.parent.name)}")
                mod_title = scopy_mod(xml_file.parent, mo_mod_folder)
                mo_mod_folder = mo_mod_path / mod_title
                (xml_file.parent / f"l{id}.manifest").write_text("")
                qInfo(f"Added mod {ascii(xml_file.parent.name)}")
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
                open(xml_file.parent / "project_file" / f"l{id}.manifest", "w+").write("")

    def mappings(self) -> List[mobase.Mapping]:
        # save mapping
        # save_mapping:List[mobase.Mapping] = []
        # if self._organizer.profile().localSavesEnabled():
        #     qInfo((self.savesDirectory().absolutePath()))
        #     save_mapping.append(mobase.Mapping(
        #         self._organizer.profile().absolutePath()+"/saves",
        #         self.savesDirectory().absolutePath(),
        #         True,
        #         True
        #         ))

        # merge mod xml
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

        # merge raid_settings.json
        override_script_path = self._get_overwrite_path() / "scripts"
        raid_settings_keys = ["torch_settings_data_table", "raid_rules_override_data_table"]
        raid_settings = json.loads(open(self._get_game_path() / "scripts" / "raid_settings.json").read())
        mod_titles = self._organizer.modList().allModsByProfilePriority()
        mod_titles = [i for i in mod_titles if self._organizer.modList().state(i) & mobase.ModState.ACTIVE]

        if not override_script_path.exists():
            override_script_path.mkdir(exist_ok=True)

        for mod_title in mod_titles:
            raid_settings_file = self._get_mo_mods_path() / mod_title / "scripts" / "raid_settings.json"
            if raid_settings_file.exists():
                try:
                    mod_raid_settings = json.loads(open(raid_settings_file).read())
                except json.JSONDecodeError:
                    continue
                for key in raid_settings_keys:
                    try:
                        raid_settings[key] += mod_raid_settings[key]
                    except KeyError:
                        continue

        open(override_script_path / "raid_settings.json", "w+").write(json.dumps(raid_settings, indent=4))

        raid_settings_mapping = [
            mobase.Mapping(
                str(override_script_path / "raid_settings.json"),
                str(self._get_game_path() / "scripts" / "raid_settings.json"),
                False,
                True,
            ),
        ]

        # merge effect files
        effect_mapping: List[mobase.Mapping] = []
        effect_files: Dict[str, List[Path]] = defaultdict(list)
        overwrite_effect_folder = self._get_overwrite_path() / "effects"

        if not overwrite_effect_folder.exists():
            overwrite_effect_folder.mkdir()
        else:
            for file in overwrite_effect_folder.glob("*.effects.darkest"):
                file.unlink()

        for mod_title in mod_titles:
            for effect_file in (self._get_mo_mods_path() / mod_title / "effects").glob("*.effects.darkest"):
                effect_files[effect_file.name].append(effect_file)

        for effect_file_name, files in effect_files.items():
            if len(files) > 1:
                contents = "\n".join([self.try_read_text(i) for i in files])
                open(overwrite_effect_folder / f"{effect_file_name}", "w+").write(contents)
                effect_mapping.append(
                    mobase.Mapping(
                        str(overwrite_effect_folder / f"{effect_file_name}"),
                        str(self._get_game_path() / "effects" / f"{effect_file_name}"),
                        False,
                        True,
                    )
                )

        # mapping static resource files
        static_resource_mapping: List[mobase.Mapping] = []
        static_resource_path = [
            "fe_flow",
            "fonts",
            "localization",
            "cursors",
            "overlays",
        ]
        for mod_title in mod_titles:
            for path in set(static_resource_path) & set([i.name for i in (self._get_mo_mods_path() / mod_title).glob("*")]):
                static_resource_mapping.append(
                    mobase.Mapping(
                        str(self._get_mo_mods_path() / mod_title / path),
                        str(self._get_game_path() / path),
                        True,
                        True,
                    )
                )

        return effect_mapping + raid_settings_mapping + static_resource_mapping

    def executables(self):
        if self.is_steam():
            path = QFileInfo(self.gameDirectory(), "_windows/darkest.exe")
        else:
            path = QFileInfo(self.gameDirectory(), "_windowsnosteam/darkest.exe")
        return [
            mobase.ExecutableInfo("Darkest Dungeon", path).withWorkingDirectory(self.gameDirectory()),
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
        documentsSaves = QDir("{}/Darkest".format(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)))
        if self.is_steam():
            cloudSaves = self.getCloudSaveDirectory()
            if cloudSaves is None:
                return documentsSaves
            return QDir(cloudSaves)
        return documentsSaves

    def listSaves(self, folder: QDir) -> list[mobase.ISaveGame]:
        folder = self.savesDirectory()
        profiles: list[Path] = []
        for path in Path(folder.absolutePath()).glob("profile_*"):
            qInfo(f"Found profile: {path.name}")
            # profile_9 is only for the Multiplayer DLC "The Butcher's Circus"
            # and contains different files than other profiles
            if path.name == "profile_9":
                continue
            profiles.append(path)

        return [DarkestDungeonSaveGame(path) for path in profiles]
