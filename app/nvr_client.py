"""
Hikvision NVR ISAPI client.

Протестировано на: Hikvision NVR, прошивка V4.x
Авторизация: Digest Auth (MD5)
Рабочие методы: POST /ISAPI/ContentMgmt/search, POST /ISAPI/ContentMgmt/download
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from requests.auth import HTTPDigestAuth

from app.config.settings import NvrSettings

from collections.abc import Callable

logger = logging.getLogger(__name__)

SEARCH_PAGE_SIZE: int = 100
DOWNLOAD_CHUNK_SIZE: int = 1024 * 1024
_NS = "http://www.hikvision.com/ver20/XMLSchema"


class NvrApiError(Exception):
    """ISAPI вернул XML-ошибку или сетевая ошибка завернута в единый тип."""

    def __init__(
        self,
        status_code: int,
        status_string: str,
        sub_status: str,
        raw_xml: str,
    ):
        self.status_code = status_code
        self.status_string = status_string
        self.sub_status = sub_status
        self.raw_xml = raw_xml
        super().__init__(
            f"ISAPI error {status_code}: {status_string} / {sub_status}"
        )


@dataclass
class ChannelInfo:
    channel_id: int
    track_id: str
    name: str
    online: bool
    ip: str | None = None


@dataclass
class RecordingSegment:
    track_id: str
    start_time: datetime
    end_time: datetime
    playback_uri: str
    codec_type: str
    size_bytes: int | None


class HikvisionNvrClient:
    """
    Клиент для работы с Hikvision NVR через ISAPI (Digest Auth).

    Пример::

        client = HikvisionNvrClient(
            host="192.168.1.64",
            username="admin",
            password="secret",
            connect_timeout_sec=30,
            read_timeout_search_sec=120,
            read_timeout_download_sec=900,
        )
        segments = client.search_recordings(
            track_id="1601",
            start_time=datetime(2026, 4, 26, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 26, 23, 59, 59, tzinfo=timezone.utc),
        )
        for seg in segments:
            client.download_segment(seg, dest_dir=Path("data/raw"))
    """

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        *,
        port: int = 80,
        connect_timeout_sec: float = 30.0,
        read_timeout_search_sec: float = 60.0,
        read_timeout_download_sec: float = 900.0,
    ):
        self.base_url = f"http://{host}:{port}"
        self._auth = HTTPDigestAuth(username, password)
        self._timeout_search = (
            float(connect_timeout_sec),
            float(read_timeout_search_sec),
        )
        self._timeout_download = (
            float(connect_timeout_sec),
            float(read_timeout_download_sec),
        )
        self._session = requests.Session()
        self._session.auth = self._auth

    @classmethod
    def from_nvr_settings(cls, nvr: NvrSettings) -> HikvisionNvrClient:
        return cls(
            host=nvr.host,
            username=nvr.username,
            password=nvr.password,
            port=nvr.port,
            connect_timeout_sec=nvr.connect_timeout_sec,
            read_timeout_search_sec=nvr.read_timeout_search_sec,
            read_timeout_download_sec=nvr.read_timeout_download_sec,
        )

    def probe_channels(self) -> list[ChannelInfo]:
        """Опрос каналов NVR через ISAPI InputProxy."""
        url_cfg = f"{self.base_url}/ISAPI/ContentMgmt/InputProxy/channels"
        url_status = f"{self.base_url}/ISAPI/ContentMgmt/InputProxy/channels/status"

        ns = {"ns": _NS}
        statuses: dict[int, bool] = {}
        try:
            r_st = self._session.get(url_status, timeout=self._timeout_search)
            if r_st.status_code == 200:
                root_st = ET.fromstring(r_st.text)
                for item in root_st.findall(".//ns:InputProxyChannelStatus", ns) or root_st.findall(".//InputProxyChannelStatus"):
                    cid_text = item.findtext("ns:id", namespaces=ns) or item.findtext("id")
                    online_text = item.findtext("ns:online", namespaces=ns) or item.findtext("online")
                    if cid_text:
                        statuses[int(cid_text)] = (str(online_text).lower() == "true")
        except Exception as exc:
            logger.warning("probe_channels status check failed: %s", exc)

        channels: list[ChannelInfo] = []
        try:
            r_cfg = self._session.get(url_cfg, timeout=self._timeout_search)
            if r_cfg.status_code == 200:
                root_cfg = ET.fromstring(r_cfg.text)
                for item in root_cfg.findall(".//ns:InputProxyChannel", ns) or root_cfg.findall(".//InputProxyChannel"):
                    cid_text = item.findtext("ns:id", namespaces=ns) or item.findtext("id")
                    if not cid_text:
                        continue
                    cid = int(cid_text)
                    name = item.findtext("ns:name", namespaces=ns) or item.findtext("name") or f"Camera {cid:03d}"
                    ip = item.findtext(".//ns:ipAddress", namespaces=ns) or item.findtext(".//ipAddress")
                    track_id = f"{cid * 100 + 1}"
                    online = statuses.get(cid, True)
                    channels.append(
                        ChannelInfo(
                            channel_id=cid,
                            track_id=track_id,
                            name=name.strip(),
                            online=online,
                            ip=ip.strip() if ip else None,
                        )
                    )
        except Exception as exc:
            logger.warning("probe_channels config check failed: %s", exc)

        return channels

    def search_recordings(
        self,
        track_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[RecordingSegment]:
        all_segments: list[RecordingSegment] = []
        position = 0
        search_id = str(uuid.uuid4()).upper()

        while True:
            logger.debug(
                "search_recordings: track=%s pos=%d start=%s end=%s",
                track_id,
                position,
                start_time.isoformat(),
                end_time.isoformat(),
            )

            xml_body = self._build_search_xml(
                search_id=search_id,
                track_id=track_id,
                start_time=start_time,
                end_time=end_time,
                max_results=SEARCH_PAGE_SIZE,
                position=position,
            )

            raw_response = self._post_xml(
                "/ISAPI/ContentMgmt/search", xml_body
            )

            segments, has_more = self._parse_search_result(raw_response)
            all_segments.extend(segments)

            logger.info(
                "search_recordings: got %d segments (total so far: %d), "
                "has_more=%s",
                len(segments),
                len(all_segments),
                has_more,
            )

            if not has_more:
                break

            position += SEARCH_PAGE_SIZE

        return all_segments

    def download_segment(
        self,
        segment: RecordingSegment,
        dest_dir: Path,
        filename: str | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            ts_start = segment.start_time.strftime("%Y%m%dT%H%M%SZ")
            ts_end = segment.end_time.strftime("%Y%m%dT%H%M%SZ")
            filename = (
                f"track{segment.track_id}_{ts_start}_{ts_end}.mp4"
            )

        dest_path = dest_dir / filename

        if dest_path.exists():
            logger.info(
                "download_segment: file already exists, skipping: %s",
                dest_path,
            )
            return dest_path

        xml_body = self._build_download_xml(segment.playback_uri)

        logger.info(
            "download_segment: start downloading %s -> %s (size hint: %s bytes)",
            segment.playback_uri[:80],
            dest_path,
            segment.size_bytes or "unknown",
        )

        url = f"{self.base_url}/ISAPI/ContentMgmt/download"

        try:
            response = self._session.post(
                url,
                data=xml_body.encode("utf-8"),
                headers={
                    "Content-Type": "application/xml",
                    "Accept": "*/*",
                },
                timeout=self._timeout_download,
                stream=True,
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "xml" in content_type.lower():
                raw_xml = response.text
                logger.error(
                    "download_segment: got XML instead of binary: %s",
                    raw_xml,
                )
                self._raise_if_error(raw_xml)

            downloaded_bytes = 0
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        if progress_callback is not None:
                            progress_callback(len(chunk))

            logger.info(
                "download_segment: done, saved %d bytes -> %s",
                downloaded_bytes,
                dest_path,
            )
            return dest_path

        except NvrApiError:
            raise
        except requests.RequestException as exc:
            if dest_path.exists():
                dest_path.unlink()
            raise NvrApiError(
                status_code=-1,
                status_string=str(exc),
                sub_status="networkError",
                raw_xml="",
            ) from exc

    def download_all_segments(
        self,
        segments: list[RecordingSegment],
        dest_dir: Path,
    ) -> list[Path]:
        results: list[Path] = []
        for i, segment in enumerate(segments, 1):
            logger.info(
                "download_all_segments: %d/%d track=%s %s -> %s",
                i,
                len(segments),
                segment.track_id,
                segment.start_time.isoformat(),
                segment.end_time.isoformat(),
            )
            try:
                path = self.download_segment(segment, dest_dir)
                results.append(path)
            except NvrApiError as exc:
                logger.error(
                    "download_all_segments: segment %d/%d failed: %s",
                    i,
                    len(segments),
                    exc,
                )
        return results

    @staticmethod
    def _build_search_xml(
        search_id: str,
        track_id: str,
        start_time: datetime,
        end_time: datetime,
        max_results: int,
        position: int,
    ) -> str:
        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        return (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            "<CMSearchDescription>\n"
            f"    <searchID>{search_id}</searchID>\n"
            "    <trackList>\n"
            f"        <trackID>{track_id}</trackID>\n"
            "    </trackList>\n"
            "    <timeSpanList>\n"
            "        <timeSpan>\n"
            f"            <startTime>{start_str}</startTime>\n"
            f"            <endTime>{end_str}</endTime>\n"
            "        </timeSpan>\n"
            "    </timeSpanList>\n"
            f"    <maxResults>{max_results}</maxResults>\n"
            f"    <searchResultPostion>{position}</searchResultPostion>\n"
            "    <metadataList>\n"
            "        <metadataDescriptor>"
            "//recordType.meta.std-cgi.com"
            "</metadataDescriptor>\n"
            "    </metadataList>\n"
            "</CMSearchDescription>\n"
        )

    @staticmethod
    def _build_download_xml(playback_uri: str) -> str:
        """XML для POST download. Текст в playbackURI записывается как есть."""

        root = ET.Element("downloadRequest")
        uri_el = ET.SubElement(root, "playbackURI")
        uri_el.text = playback_uri
        return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(
            root,
            encoding="unicode",
        )

    def _post_xml(self, path: str, body: str) -> str:
        url = f"{self.base_url}{path}"
        try:
            response = self._session.post(
                url,
                data=body.encode("utf-8"),
                headers={
                    "Content-Type": "application/xml",
                    "Accept": "application/xml",
                },
                timeout=self._timeout_search,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise NvrApiError(
                status_code=-1,
                status_string=str(exc),
                sub_status="networkError",
                raw_xml="",
            ) from exc

        self._raise_if_error(response.text)
        return response.text

    def _parse_search_result(
        self, raw_xml: str
    ) -> tuple[list[RecordingSegment], bool]:
        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError as exc:
            raise NvrApiError(-1, "XML parse error", str(exc), raw_xml) from exc

        ns = {"h": _NS}

        response_strg = self._find_text(
            root, "h:responseStatusStrg", ns, default="OK"
        )
        has_more = response_strg.upper() == "MORE"

        segments: list[RecordingSegment] = []

        for item in root.findall(".//h:searchMatchItem", ns):
            try:
                segment = self._parse_match_item(item, ns)
                segments.append(segment)
            except Exception as exc:
                logger.warning(
                    "_parse_search_result: skipping item due to error: %s",
                    exc,
                )

        return segments, has_more

    @staticmethod
    def _parse_match_item(
        item: ET.Element, ns: dict
    ) -> RecordingSegment:
        track_id = HikvisionNvrClient._find_text(
            item, "h:trackID", ns, required=True
        )

        start_str = HikvisionNvrClient._find_text(
            item, "h:timeSpan/h:startTime", ns, required=True
        )
        end_str = HikvisionNvrClient._find_text(
            item, "h:timeSpan/h:endTime", ns, required=True
        )

        playback_uri = HikvisionNvrClient._find_text(
            item,
            "h:mediaSegmentDescriptor/h:playbackURI",
            ns,
            required=True,
        )
        codec_type = HikvisionNvrClient._find_text(
            item,
            "h:mediaSegmentDescriptor/h:codecType",
            ns,
            default="unknown",
        )

        size_bytes: int | None = None
        if "size=" in playback_uri:
            try:
                size_part = playback_uri.split("size=")[-1].split("&")[0]
                size_bytes = int(size_part)
            except ValueError:
                pass

        return RecordingSegment(
            track_id=track_id,
            start_time=datetime.strptime(
                start_str,
                "%Y-%m-%dT%H:%M:%SZ",
            ).replace(tzinfo=timezone.utc),
            end_time=datetime.strptime(end_str, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            ),
            playback_uri=playback_uri,
            codec_type=codec_type,
            size_bytes=size_bytes,
        )

    @staticmethod
    def _raise_if_error(raw_xml: str) -> None:
        if "<ResponseStatus" not in raw_xml and "<responseStatus>" not in raw_xml:
            return

        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError:
            return

        tag_lc = root.tag.lower()
        if "responsestatus" not in tag_lc:
            return

        ns = {"h": _NS}
        status_code_str = HikvisionNvrClient._find_text(
            root, "h:statusCode", ns
        ) or HikvisionNvrClient._find_text(root, "statusCode", {})
        status_string = (
            HikvisionNvrClient._find_text(root, "h:statusString", ns)
            or HikvisionNvrClient._find_text(root, "statusString", {})
            or ""
        )
        sub_status = (
            HikvisionNvrClient._find_text(root, "h:subStatusCode", ns)
            or HikvisionNvrClient._find_text(root, "subStatusCode", {})
            or ""
        )

        try:
            code = int(status_code_str)
        except (TypeError, ValueError):
            code = -1

        if code != 1:
            logger.error(
                "_raise_if_error: ISAPI error: code=%s string=%s sub=%s xml=%s",
                code,
                status_string,
                sub_status,
                raw_xml,
            )
            raise NvrApiError(code, status_string, sub_status, raw_xml)

    @staticmethod
    def _find_text(
        element: ET.Element,
        path: str,
        ns: dict,
        default: str = "",
        required: bool = False,
    ) -> str:
        found = element.find(path, ns)
        if found is None or found.text is None:
            if required:
                raise ValueError(f"Required XML element not found: {path}")
            return default
        return found.text.strip()


def _client_from_env_for_smoke(host: str, port: int) -> HikvisionNvrClient:
    import os

    username = (os.environ.get("NVR__USERNAME") or "").strip()
    password = (os.environ.get("NVR__PASSWORD") or "").strip()
    if not username or not password:
        raise SystemExit(
            "Задайте NVR__USERNAME и NVR__PASSWORD в окружении для smoke-проверки."
        )

    return HikvisionNvrClient(
        host=host,
        username=username,
        password=password,
        port=port,
        connect_timeout_sec=30,
        read_timeout_search_sec=120,
        read_timeout_download_sec=900,
    )


if __name__ == "__main__":
    import logging as _logging
    import os as _os
    from datetime import datetime as _dt, timezone as _tz

    _logging.basicConfig(level=_logging.DEBUG)

    _host = _os.environ.get("NVR_SMOKE_HOST", "192.168.1.64")
    _port = int(_os.environ.get("NVR_SMOKE_PORT", "80"))
    _track = _os.environ.get("NVR_SMOKE_TRACK", "1601")

    client = _client_from_env_for_smoke(_host, _port)
    segs = client.search_recordings(
        track_id=_track,
        start_time=_dt(
            2026, 4, 26, 0, 0, 0, tzinfo=_tz.utc
        ),
        end_time=_dt(
            2026,
            4,
            26,
            23,
            59,
            59,
            tzinfo=_tz.utc,
        ),
    )
    print(f"Found {len(segs)} segments")
    for s in segs:
        print(f"  {s.start_time} -> {s.end_time}  size={s.size_bytes}")
