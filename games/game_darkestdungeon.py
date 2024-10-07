from collections import defaultdict
import json
from pathlib import Path
import random
import re
import shutil
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element
from typing import Dict, Iterable, List  # type: ignore
import mobase
from PyQt6.QtCore import QDir, QFileInfo, QStandardPaths, qInfo

from ..basic_game import BasicGame, BasicGameSaveGame
from ..steam_utils import find_steam_path, find_games


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
            return cls(
                mod_title, mod_versions, mod_tags, mod_description, mod_PublishedFileId
            )
        root = tree.getroot()
        mod_title = cls.etree_text_iter(root, "Title") or mod_title
        mod_title = re.sub(r'[\/:*?"<>|]', "_", mod_title).strip()
        mod_versions[0] = int(
            cls.etree_text_iter(root, "VersionMajor") or mod_versions[0]
        )
        mod_versions[1] = int(
            cls.etree_text_iter(root, "VersionMinor") or mod_versions[1]
        )
        mod_versions[2] = int(
            cls.etree_text_iter(root, "TargetBuild") or mod_versions[2]
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
        for icon, index in zip(icons, range(1, 1 + len(icons))):
            pass
            self.modIdMap[icon.parent.parent.name] = index
            contents.append(mobase.ModDataContent.Content(index, "缩略图", str(icon)))
        icons = [i for i in Path(self.modPath).glob("*/preview_icon.png")]
        for icon, index in zip(icons, range(1, 1 + len(icons))):
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
            return magic == b"/x01/xb1/x00/x00"

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
    GameSupportURL = (
        r"https://github.com/ModOrganizer2/modorganizer-basic_games/wiki/"
        "Game:-Darkest-Dungeon"
    )

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

    def _get_game_path(self):
        return Path(self.gameDirectory().absolutePath())

    def _get_mods_path(self):
        return Path(self._organizer.modsPath())

    def Refreshed(self):
        pass

        def acf_parser(
            acf_file: str,
        ) -> Dict[str, Dict[str, Dict[str, Dict[str, str]]]]:
            acf_content = open(acf_file).read()
            acf_content = re.sub(r'(".*?")\t*', r"\g<1>:", acf_content)
            acf_content = "{" + acf_content + "}"
            acf_content = re.sub(r':(\n\t*")', r",\g<1>", acf_content)
            acf_content = re.sub(r":(\n\t*\})", r",\g<1>", acf_content)
            acf_content = re.sub(r"\}", r"},", acf_content)
            acf_content = re.sub(r",(\n\t*\})", r"\g<1>", acf_content)
            acf_content = acf_content.strip(",")
            return json.loads(acf_content)

        def copy_local_mod(scr: str | Path, dst: str | Path) -> str:  # type: ignore
            scr = str(scr)
            dst = str(dst)
            if not Path(scr).exists():
                return ""
            if Path(dst).exists():
                return copy_local_mod(scr, dst + "_copy")  # type: ignore
            else:
                shutil.copytree(scr, dst)
                return dst

        qInfo("refreshing")
        acf_path = (
            find_games()["262060"].parent.parent / "workshop" / "appworkshop_262060.acf"
        )
        workshop_items = acf_parser(str(acf_path))["AppWorkshop"]["WorkshopItemDetails"]
        qInfo("found " + str(len(workshop_items)) + " items in workshop.")
        modlist = self._organizer.modList()
        mod_root_folder = Path(self._organizer.modsPath())
        mod_names = modlist.allMods()
        mo_workshop_PublishedFileId: Dict[str, mobase.IModInterface] = {
            str(i.stem.strip("w")): modlist.getMod(str(i.parent.parent.name))
            for i in mod_root_folder.glob("*/project_file/w*.manifest")
        }
        mo_local_PublishedFileId: Dict[str, mobase.IModInterface] = {
            str(i.stem.strip("l")): modlist.getMod(str(i.parent.parent.name))
            for i in mod_root_folder.glob("*/project_file/l*.manifest")
        }

        for mod_name in mod_names:
            mod = modlist.getMod(mod_name)
            mod_folder = mod_root_folder / mod_name
            preview_file = mod_folder / "preview_icon.png"
            xml_file = [i for i in (mod_folder / "project_file").glob("*.xml")] + [
                i for i in mod_folder.glob("project.xml")
            ]
            xml_file = xml_file[0] if xml_file else Path("None")
            manifest_file = [
                i for i in (mod_folder / "project_file").glob("*.manifest")
            ]
            manifest_file = manifest_file[0] if manifest_file else Path("None")
            txt_file = mod_folder / "modfiles.txt"
            log_file = mod_folder / "steam_workshop_uploader.log"

            if log_file.exists():
                log_file.unlink()

            if not xml_file.exists():
                continue
            mod_data = xml_data.mod_xml_parser(xml_file)

            # set version
            if not mod.version().isValid():
                qInfo(
                    f"Setting version of {ascii(mod_name)} to {mod_data.mod_versions}"
                )
                mod.setVersion(mobase.VersionInfo(*mod_data.mod_versions))
                qInfo(f"Set version of {ascii(mod_name)} to {mod_data.mod_versions}")

            # add categories
            if not mod.categories():
                qInfo(f"Adding category to {ascii(mod_name)}")
                for i in mod_data.mod_tags:
                    if not i in mod.categories():
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
                qInfo(f"Added category to {ascii(mod_name)}")

            # set url
            if (
                not mod.url()
                and manifest_file.exists()
                and manifest_file.name.startswith("w")
            ):
                qInfo(
                    f"Setting url of {ascii(mod_name)} to {mod_data.mod_PublishedFileId}"
                )
                mod.setUrl(
                    f"https://steamcommunity.com/sharedfiles/filedetails/?id={mod_data.mod_PublishedFileId}"
                )
                qInfo(f"Set url of {ascii(mod_name)} to {mod_data.mod_PublishedFileId}")

        # copy steam workshop mods
        # 使用mod_PublishedFileId保证唯一，可能会出现本地mod占用的问题
        workshop_path = (
            find_games()["262060"].parent.parent / "workshop" / "content" / "262060"
        )
        for PublishedFileId in set([i for i in workshop_items.keys()]) - set(
            [i for i in mo_workshop_PublishedFileId.keys()]
        ):
            mod_title = xml_data.mod_xml_parser(
                workshop_path / PublishedFileId / "project.xml"
            ).mod_title
            qInfo(f"Adding mod {ascii(mod_title)}")
            if mod_title := copy_local_mod(
                workshop_path / PublishedFileId, mod_root_folder / mod_title
            ):
                mo_mod_folder = mod_root_folder / mod_title
                preview_file = mo_mod_folder / "preview_icon.png"
                txt_file = mo_mod_folder / "modfiles.txt"
                log_file = mo_mod_folder / "steam_workshop_uploader.log"
                xml_file = mo_mod_folder / "project.xml"
                if preview_file.exists():
                    (mo_mod_folder / "preview_file").mkdir(exist_ok=True)
                    preview_file.rename(
                        mo_mod_folder / "preview_file" / (PublishedFileId + ".png")
                    )
                if txt_file.exists():
                    txt_file.unlink()
                if log_file.exists():
                    log_file.unlink()
                (mo_mod_folder / "project_file").mkdir(exist_ok=True)
                if xml_file.exists():
                    xml_file.rename(
                        mo_mod_folder / "project_file" / (PublishedFileId + ".xml")
                    )
                    open(
                        mo_mod_folder
                        / "project_file"
                        / ("w" + PublishedFileId + ".manifest"),
                        "w+",
                    ).write(workshop_items[PublishedFileId]["manifest"])
            else:
                qInfo(f"Failed to add mod {ascii(mod_title)}")
            qInfo(f"Added mod {ascii(mod_title)}")
        for PublishedFileId in set([i for i in workshop_items.keys()]) & set(
            [i for i in mo_workshop_PublishedFileId]
        ):
            mod = modlist.getMod(PublishedFileId)
            if not mod:
                continue
            new_manifest = workshop_items[PublishedFileId]["manifest"]
            mo_workshop_PublishedFileId[PublishedFileId].absolutePath()
            old_manifest_file = (
                Path(mo_workshop_PublishedFileId[PublishedFileId].absolutePath())
                / "project_file"
                / ("w" + PublishedFileId + ".manifest")
            )
            old_manifest = (
                old_manifest_file.read_text() if old_manifest_file.exists() else ""
            )
            if not old_manifest:
                qInfo(
                    mo_workshop_PublishedFileId[PublishedFileId].name()
                    + " manifest is empty"
                )
            if new_manifest != old_manifest:
                mo_workshop_PublishedFileId[PublishedFileId].setNewestVersion(
                    mobase.VersionInfo(
                        *xml_data.mod_xml_parser(
                            workshop_path / PublishedFileId / "project.xml"
                        ).mod_versions
                    )
                )

        # copy local mods
        for xml_file in (Path(self.gameDirectory().absolutePath()) / "mods").glob(
            "*/project.xml"
        ):
            mod_xml_data = xml_data.mod_xml_parser(xml_file)
            local_mod_folder = xml_file.parent
            mo_mod_folder = mod_root_folder / mod_xml_data.mod_title
            manifest_files = [i for i in local_mod_folder.glob("*.manifest")]
            id = str(random.randint(1, 99999))
            set([i.stem.strip("l") for i in manifest_files])
            if not manifest_files or not (
                set([i.stem.strip("l") for i in manifest_files])
                & set([i for i in mo_local_PublishedFileId.keys()])
            ):
                qInfo(f"Adding mod {ascii(xml_file.parent.name)}")
                mod_title = copy_local_mod(xml_file.parent, mo_mod_folder)
                mo_mod_folder = mod_root_folder / mod_title
                qInfo(f"Added mod {ascii(xml_file.parent.name)}")
                (local_mod_folder / ("l" + id + ".manifest")).write_text("")
            else:
                continue

            preview_file = mo_mod_folder / "preview_icon.png"
            txt_file = mo_mod_folder / "modfiles.txt"
            xml_file = mo_mod_folder / "project.xml"
            if preview_file.exists():
                (mo_mod_folder / "preview_file").mkdir(exist_ok=True)
                preview_file.rename(mo_mod_folder / "preview_file" / (id + ".png"))
            if txt_file.exists():
                txt_file.unlink()
            (mo_mod_folder / "project_file").mkdir(exist_ok=True)
            if xml_file.exists():
                xml_file.rename(xml_file.parent / "project_file" / (id + ".xml"))
                open(
                    xml_file.parent / "project_file" / ("l" + id + ".manifest"), "w+"
                ).write("")

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
        project_xml = Path(self._organizer.overwritePath()) / "project.xml"
        if not project_xml.exists():
            project_xml.touch()
        project_xml.write_text(project_text)

        # merge raid_settings.json
        override_script_path = Path(self._organizer.overwritePath()) / "scripts"
        raid_settings_keys = [
            "torch_settings_data_table",
            "raid_rules_override_data_table",
        ]
        raid_settings = json.loads(
            open(self._get_game_path() / "scripts" / "raid_settings.json").read()
        )
        modlist = self._organizer.modList().allModsByProfilePriority()
        modlist = [
            i
            for i in modlist
            if self._organizer.modList().state(i) & mobase.ModState.ACTIVE
        ]
        for mod in modlist:
            raid_settings_file = (
                self._get_mods_path() / mod / "scripts" / "raid_settings.json"
            )
            if raid_settings_file.exists():
                for key in raid_settings_keys:
                    raid_settings[key] += json.loads(open(raid_settings_file).read())[
                        key
                    ]
        if not override_script_path.exists():
            override_script_path.mkdir(exist_ok=True)
        open(override_script_path / "raid_settings.json", "w+").write(
            json.dumps(raid_settings, indent=4)
        )
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
        if not (Path(self._organizer.overwritePath()) / "effects").exists():
            (Path(self._organizer.overwritePath()) / "effects").mkdir(exist_ok=True)
        else:
            for file in (Path(self._organizer.overwritePath()) / "effects").glob(
                "*.effects.darkest"
            ):
                file.unlink()
        for mod in modlist:
            for effect_file in (self._get_mods_path() / mod / "effects").glob(
                "*.effects.darkest"
            ):
                effect_files[effect_file.name].append(effect_file)
        for key, files in effect_files.items():
            if len(files) > 1:
                contents = "\n".join([i.read_text() for i in files])
                open(
                    Path(self._organizer.overwritePath()) / "effects" / f"{key}", "w+"
                ).write(contents)
                effect_mapping.append(
                    mobase.Mapping(
                        str(
                            Path(self._organizer.overwritePath()) / "effects" / f"{key}"
                        ),
                        str(self._get_game_path() / "effects" / f"{key}"),
                        False,
                        True,
                    )
                )

        # mapping other files
        nomod_mapping: List[mobase.Mapping] = []
        none_mod_dirs = ["fe_flow", "fonts", "localization", "cursors", "overlays"]
        for mod in modlist:
            for dir in set(none_mod_dirs) & set(
                [i.name for i in (self._get_mods_path() / mod).glob("*")]
            ):
                qInfo(
                    f"mapping {ascii(str(self._get_mods_path() / mod / dir))} to {ascii(str(self._get_game_path() / dir))}"
                )
                nomod_mapping.append(
                    mobase.Mapping(
                        str(self._get_mods_path() / mod / dir),
                        str(self._get_game_path() / dir),
                        True,
                        True,
                    )
                )

        return effect_mapping + raid_settings_mapping + nomod_mapping

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
        qInfo(f"{folder.absolutePath()}")
        profiles: list[Path] = []
        for path in Path(folder.absolutePath()).glob("profile_*"):
            qInfo(f"Found profile: {path.name}")
            # profile_9 is only for the Multiplayer DLC "The Butcher's Circus"
            # and contains different files than other profiles
            if path.name == "profile_9":
                continue
            profiles.append(path)

        return [DarkestDungeonSaveGame(path) for path in profiles]
