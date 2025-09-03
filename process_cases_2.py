# encoding: utf-8
from datetime import datetime
import re
from typing import Dict, List, Optional, Tuple
from const import AMOUNT_PATTERN, CHN_NUM_CHARS, EDU_LEVELS, SEVERITY_TERMS, CRIME_MAP

import cn2an as cn2an_convert
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
import cpca

# 读取csv


def try_read_csv(path: str):
    encodings_to_try = [
        "utf-8",
        "utf-8-sig",
        "gb18030",
        "gbk",
        "utf-16",
    ]
    last_err = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df, enc
        except Exception as e:  # noqa: BLE001 - 宽松兼容外部文件错误
            last_err = e
            continue
    raise RuntimeError(f"无法读取CSV，请检查编码或文件格式。最后错误: {last_err}")

# 文本预处理


def normalize_text(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    # 统一全角空格、去掉多余空白
    normalized = re.sub(r"\u3000", " ", text)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()

# 金额匹配


def parse_amount_to_yuan(matched_text: str) -> Optional[float]:
    try:
        # 修复数字匹配逻辑：先匹配任意连续数字和逗号，然后处理逗号
        amount_match = re.search(
            r"([0-9,]+)(?:\.[0-9]+)?", matched_text
        )
        if not amount_match:
            return None
        number_str = amount_match.group(1).replace(",", "")
        # 确保提取的是纯数字
        if not number_str.isdigit():
            return None
        number_val = float(number_str)
        is_wan = "万" in matched_text
        return number_val * (10000.0 if is_wan else 1.0)
    except Exception:  # noqa: BLE001
        return None

# 中文数字转换成数字


def extract_involved_amount(text: str) -> Optional[float]:
    # 侧重关键词邻近，如 涉案金额/数额、价值、赃款、非法所得
    candidates: List[float] = []
    for m in re.finditer(
        r"(涉案金额|涉案数额|数额|价值|赃款|非法所得)[^。；;，,：:]{0,15}?((?:人民币)?\s*[0-9,]+(?:\.[0-9]+)?\s*(?:万)?\s*元)",
        text,
    ):
        amt = parse_amount_to_yuan(m.group(2))
        if amt is not None:
            candidates.append(amt)
    # 退而求其次，抓取所有金额取最大
    if not candidates:
        for m in AMOUNT_PATTERN.finditer(text):
            amt = parse_amount_to_yuan(m.group(0))
            if amt is not None:
                candidates.append(amt)
    return max(candidates) if candidates else None

# 罚金金额


def parse_chinese_amount_to_yuan(matched_text: str) -> Optional[float]:
    try:
        s = re.sub(r"(人民币|元|圆|正|整)", "", matched_text)
        s = s.strip()
        if not s:
            return None
        val = cn2an_convert.cn2an(s, "smart")
        return float(val)
    except Exception:  # noqa: BLE001
        return None


def extract_fine_amount(text: str) -> Optional[float]:
    candidates: List[float] = []
    for m in re.finditer(
        r"(并处|并以)?罚(金|款)[^。；;，,：:]{0,15}?((?:人民币)?\s*[0-9,]+(?:\.[0-9]+)?\s*(?:万)?\s*元)",
        text,
    ):
        amt = parse_amount_to_yuan(m.group(3))
        if amt is not None:
            candidates.append(amt)

        # 中文数字金额（如 十万元、五万元、一万五千元）
    if len(candidates) == 0:
        chn_pattern = rf"(并处|并以)?罚(金|款)[^。；;，,：:]{{0,20}}?((?:人民币)?\s*[{CHN_NUM_CHARS}]+?\s*元)"
        for m in re.finditer(chn_pattern, text):
            amt = parse_chinese_amount_to_yuan(m.group(3))
            if amt is not None:
                candidates.append(amt)
    return max(candidates) if candidates else None

# 犯罪人的年龄


def extract_age(text: str) -> Optional[int]:
    # 直接年龄
    m = re.search(r"(?<![0-9])([1-9][0-9])\s*岁", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:  # noqa: BLE001
            pass
    # "年龄为/是 X 岁"
    m = re.search(r"年龄[为是]?\s*([1-9][0-9])\s*岁?", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:  # noqa: BLE001
            pass
    return None

# 受教育程度


def extract_education(text: str) -> Optional[str]:
    for lvl in EDU_LEVELS:
        if lvl in text:
            return lvl
    # 诸如 "受教育程度为X" 的表达
    m = re.search(r"受教育程度[为是]\s*([\u4e00-\u9fa5]{1,6})", text)
    if m:
        return m.group(1)
    return None

# 地点属性


def extract_location(text: str) -> Optional[str]:
    # 优先依据显式提示词抓取包含行政区后缀的短片段
    hint = re.search(
        r"(案发地|案发地点|地点|住址|住所|户籍地|籍贯|来源地|被告人来源地)[^。；;：:]{0,20}",
        text,
    )
    target_span = text
    if hint:
        target_span = hint.group(0)
    m = re.search(r"([\u4e00-\u9fa5]{2,20}?(?:省|市|区|县|乡|镇|村))", target_span)
    if m:
        return m.group(1)
    # 退而求其次，全局抓一个带行政区后缀的短地名
    m = re.search(r"([\u4e00-\u9fa5]{2,12}(?:省|市|区|县))", text)
    if m:
        return m.group(1)
    return None

# 案件严重程度


def extract_severity(text: str) -> Optional[str]:
    hits = []
    for term in SEVERITY_TERMS:
        if term in text:
            hits.append(term)
    # 根据命中程度排序，只取最严重的
    if hits:
        return sorted(set(hits), key=lambda x: SEVERITY_TERMS.index(x))[0]
    return None

# 案件地区


def extract_area(text: str, address: str, area_type: str) -> Optional[str]:
    # 归一化输入，防止 float/NaN 触发正则报错
    addr = address if isinstance(address, str) else ""
    txt = text if isinstance(text, str) else ""
    patterns = {
        "省": r"([\u4e00-\u9fa5]{1,20}省)",
        # "自治区": r"([\u4e00-\u9fa5]{1,20}自治区|[\u4e00-\u9fa5]{1,20}特别行政区)",
        "市": r"([\u4e00-\u9fa5]{1,20}市)",
        "县": r"([\u4e00-\u9fa5]{1,20}(?:自治县|县))",
        "区": r"([\u4e00-\u9fa5]{1,20}区)",
        # "旗": r"([\u4e00-\u9fa5]{1,20}旗)",
    }
    # 默认回退：匹配以 area_type 结尾的中文地名
    pattern = patterns.get(
        area_type, rf"([\u4e00-\u9fa5]{{1,20}}(?:{re.escape(area_type)}))")

    for source_name, source in (("address", addr), ("text", txt)):
        if not source:
            continue
        try:
            m = re.search(pattern, source)
            if m:
                result = cpca.transform([m.group(1)])
                for row in result.to_dict(orient="records"):
                    if (area_type == '县'):
                        return row.get("区") or row.get("地址")
                    return row.get(area_type)
        except Exception as e:
            # 打印上下文，便于定位哪段文本导致失败
            snippet = source[:120]
            print(
                f"[extract_area] error area_type={area_type} source={source_name} pattern={pattern!r} snippet={snippet!r} err={e}")
            return None
    return None

# 审判时间


# def extract_trial_date(text: str) -> Optional[str]:
#     # YYYY年M月D日 或 YYYY年M月
#     date_pattern = r"([12][0-9]{3})年\s*([01]?[0-9])月(?:\s*([0-3]?[0-9])日)?"
#     # 优先靠近 判决/裁定/宣判 关键词
#     for key in ["判决", "裁定", "宣判", "审理", "开庭"]:
#         m = re.search(key + r"[^。；;：:]{0,10}" + date_pattern, text)
#         if m:
#             year, month, day = m.group(1), m.group(2), m.group(3)
#             if day:
#                 return f"{year}年{int(month)}月{int(day)}日"
#             return f"{year}年{int(month)}月"
#     m = re.search(date_pattern, text)
#     if m:
#         year, month, day = m.group(1), m.group(2), m.group(3)
#         if day:
#             return f"{year}年{int(month)}月{int(day)}日"
#         return f"{year}年{int(month)}月"
#     return None


def extract_birthday(full_text, judge_data):
    """
    提取出生年月，支持多种格式：
    - 1983年8月23日出生
    - 1980年12月3日出生
    - 1988年5月16日出生
    - 1983年8月23日出生于...
    """
    if not isinstance(full_text, str):
        return None

    # 主要模式：YYYY年M月D日出生
    patterns = [
        r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日\s*出生",
        r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日\s*出生于",
    ]

    for pattern in patterns:
        match = re.search(pattern, full_text)
        if match:
            year, month, day = match.groups()
            if judge_data:
                return calculate_age(f"{year}/{month}/{day}", judge_data)
    return None


def calculate_age(birth_date, target_date):
    """
    计算年龄，支持多种日期格式
    """
    def parse_date(date_str):
        """解析多种日期格式"""
        if not isinstance(date_str, str):
            return None
        
        # 支持的日期格式
        date_formats = [
            "%Y/%m/%d",      # 2023/1/4
            "%Y-%m-%d",      # 2021-1-3
            "%Y年%m月%d日",   # 2021年1月3日
            "%Y.%m.%d",      # 2021.1.3
            "%Y/%m/%d",      # 2023/01/04 (补零格式)
            "%Y-%m-%d",      # 2021-01-03 (补零格式)
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    # 解析日期
    birth = parse_date(birth_date)
    target = parse_date(target_date)
    
    if birth is None or target is None:
        return None
    
    # 计算年龄
    age = target.year - birth.year
    
    # 检查是否过了生日
    # 如果目标日期的月份小于出生月份，或者月份相同但日期小于出生日期
    if (target.month < birth.month) or (target.month == birth.month and target.day < birth.day):
        age -= 1  # 还没过生日，年龄减1
    
    return age if age >= 0 else None
# 检测文本和案由中的罪名关键词命中情况


def detect_hits_and_keywords(
    text: str, case_type: str, crime_map: Dict[str, List[str]]
) -> Tuple[List[str], List[str]]:
    hit_categories: List[str] = []
    hit_keywords: List[str] = []

    # 合并全文和案由内容进行检测
    combined_content = text + " " + normalize_text(case_type)
    for crime, kws in crime_map.items():
        for kw in kws:
            if kw and kw in combined_content:
                if crime not in hit_categories:
                    hit_categories.append(crime)
                hit_keywords.append(kw)
    return hit_categories, sorted(set(hit_keywords))


def process_dataframe(df) -> "DataFrame":  # type: ignore[name-defined]
    # 统一列名（容错：可能存在空格或BOM）
    df = df.rename(columns={c.strip(): c.strip() for c in df.columns})

    # 打印行数
    for col in ["案号", "案由", "全文"]:
        if col not in df.columns:
            raise ValueError(f"缺少必要列：{col}")
    crime_map = CRIME_MAP

    records = []
    for _, row in df.iterrows():
        full_text = normalize_text(row.get("全文"))
        case_type = row.get("案由", "")
        if not full_text:
            continue

        # 如果records中存在案号相同的行，则跳过
        if any(record.get("案号") == row.get("案号") for record in records):
            continue

        cats, kws = detect_hits_and_keywords(full_text, case_type, crime_map)
        if not cats:
            continue
        crime = kws[0]
        if (crime[-1] != "罪"):
            crime += "罪"
        age = extract_birthday(full_text, row.get("裁判日期"))
        edu = extract_education(full_text)
        loc = extract_location(full_text)
        amount = extract_involved_amount(full_text)
        severity = extract_severity(full_text)
        province = extract_area(full_text, row.get("所属地区"), "省")
        city = extract_area(full_text, row.get("所属地区"), "市")
        county = extract_area(full_text, row.get("所属地区"), "县")
        court = row.get("法院")
        case_type_to = row.get("案件类型")
        law_basis = row.get("法律依据")
        trial_date = row.get("裁判日期")
        public_date = row.get("公开日期")
        trial_procedure = row.get("审理程序")
        area = row.get("所属地区")
        defendant = row.get("当事人")
        fine_amount = extract_fine_amount(full_text)
        records.append(
            {
                "案号": row.get("案号"),
                "案由": row.get("案由"),
                "法院": court,
                "省": province,
                "市": city,
                "县": county,
                "审理程序": trial_procedure,
                "所属地区": area,
                # "出生年月日": birth_date,
                "犯罪人的年龄": age,
                "受教育程度": edu,
                "案件地点": loc,
                "案件严重程度": severity,
                "裁判日期": trial_date,
                "公开日期": public_date,
                "案件类型": case_type_to,
                "法律依据": law_basis,
                "当事人": defendant,
                "涉案金额_元": amount,
                "罚金金额_元": fine_amount,
                "罪名": crime,
                "全文": row.get("全文"),
            }
        )
        print("案号:", row.get("案号"), "加入记录", len(records))

    result_df = pd.DataFrame.from_records(
        records,
        columns=[
            "案号",
            "案由",
            "法院",
            "省",
            "市",
            "县",
            "审理程序",
            "所属地区",
            "犯罪人的年龄",
            # "出生年月日",
            "受教育程度",
            "案件地点",
            "案件严重程度",
            "裁判日期",
            "公开日期",
            "案件类型",
            "法律依据",
            "当事人",
            "涉案金额_元",
            "罚金金额_元",
            "罪名",
            "全文"
        ],
    )
    return result_df


def detect_csv_encoding(path: str, sample_nrows: int = 2000) -> str:
    encodings_to_try = ["utf-8", "utf-8-sig", "gb18030", "gbk", "utf-16"]
    last_err = None
    for enc in encodings_to_try:
        try:
            pd.read_csv(path, encoding=enc, nrows=sample_nrows)
            return enc
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"无法识别CSV编码。最后错误: {last_err}")


def main():
    # 路径数组
    input_paths = [
        "../2017年裁判文书数据1/2017年01月裁判文书数据.csv",
        "../2018年裁判文书数据1/2018年01月裁判文书数据.csv",
        "../2019年裁判文书数据1/2019年01月裁判文书数据.csv",
        "../2020年裁判文书数据1/2020年01月裁判文书数据.csv",
        "../2021年裁判文书数据1/2021年01月裁判文书数据.csv",
    ]
    output_paths = [
        "./输出结果/2017年数据/2017年01月裁判文书数据.csv",
        "./输出结果/2018年数据/2018年01月裁判文书数据.csv",
        "./输出结果/2019年数据/2019年01月裁判文书数据.csv",
        "./输出结果/2020年数据/2020年01月裁判文书数据.csv",
        "./输出结果/2021年数据/2021年01月裁判文书数据.csv",
    ]

    usecols = ["案号", "案由", "全文", "裁判日期", "所属地区",
               "案件类型", "法律依据", "公开日期", "当事人", "法院", "审理程序"]  # 只读必要列，加速IO与序列化
    chunksize = 100000  # 可按内存/CPU调大到 8~20万
    workers = max(1, (os.cpu_count() or 4) - 1)

    for in_rel, out_rel in zip(input_paths, output_paths):
        input_path = os.path.join(os.path.dirname(__file__), in_rel)
        output_path = os.path.join(os.path.dirname(__file__), out_rel)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)

        enc = detect_csv_encoding(input_path)

        seen_case_ids = set()
        header_written = False

        chunks = pd.read_csv(
            input_path,
            encoding=enc,
            chunksize=chunksize,
            usecols=usecols,
            dtype=str,  # 避免类型推断带来的开销/NaN陷阱
        )

        with ProcessPoolExecutor(max_workers=workers) as ex:
            for result_df in ex.map(process_dataframe, chunks, chunksize=1):
                if result_df is None or result_df.empty:
                    continue

                # 按案号去重（在当前文件范围内）
                if "案号" in result_df.columns:
                    mask = ~result_df["案号"].isin(seen_case_ids)
                    result_df = result_df[mask]
                    if not result_df.empty:
                        seen_case_ids.update(result_df["案号"].dropna().tolist())

                if not result_df.empty:
                    result_df.to_csv(
                        output_path,
                        mode="a",
                        index=False,
                        header=(not header_written),
                        encoding="utf-8",
                    )
                    header_written = True

        print(f"读取编码: {enc}")
        print(f"输入: {input_path}")
        print(f"输出: {output_path}")


if __name__ == "__main__":
    main()
