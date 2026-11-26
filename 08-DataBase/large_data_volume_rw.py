import time
import pandas as pd
from sqlalchemy import create_engine
import xlsxwriter


############################################################
# ========== 配置信息 ==========
############################################################
DB_URL = "mysql+pymysql://root:password@host:3306/dbname?charset=utf8mb4"
TABLE = "ods_production_requirement_property"
OUTPUT_FILE = "export_5000w.xlsx"
READ_CHUNK_SIZE = 50000          # 从数据库每批读取的行数，可适当调大
SHEET_ROW_LIMIT = 1_000_000      # 每个 Sheet 数据行上限（含表头约 100 万 + 1 行）

COLUMNS = [
            "ID", "REQUIREMENT_ID", "REQUIREMENT_CODE", "PROPERTY_CODE",
            "PROPERTY_NAME", "STRING_VALUE", "VALUE", "VALUE_MAX", "VALUE_MIN",
            "UNIT", "STAGE", "RANK", "VERSION", "CATEGORY", "CONDITION_SET_ID",
            "REMARK", "CREATION_TIME", "CREATOR_ID", "LAST_MODIFICATION_TIME",
            "LAST_MODIFIER_ID",
            ]

engine = create_engine(DB_URL)
sql = f"SELECT {', '.join(COLUMNS)} FROM {TABLE} ORDER BY id"

# 直接使用 xlsxwriter，constant_memory 才能真正生效
workbook = xlsxwriter.Workbook(
                                OUTPUT_FILE,
                                options={
                                            "constant_memory": True,    # 流式写入，内存占用恒定
                                            "nan_inf_to_errors": True,  # 防止 NaN/Inf 中断写入
                                        },
                                )

header_fmt = workbook.add_format({"bold": True, "bg_color": "#DDEBF7"})
datetime_fmt = workbook.add_format({"num_format": "yyyy-mm-dd hh:mm:ss"})

state = {"sheet_index": 0, "worksheet": None, "row_in_sheet": 0}


############################################################
# 创建sheet页
############################################################
def new_sheet():
    state["sheet_index"] += 1
    ws = workbook.add_worksheet(f"Data_Part_{state['sheet_index']}")
    ws.write_row(0, 0, COLUMNS, header_fmt)
    ws.freeze_panes(1, 0)
    for i, c in enumerate(COLUMNS):
        ws.set_column(i, i, max(len(c) + 2, 12))
    state["worksheet"] = ws
    state["row_in_sheet"] = 1  # 表头占第 1 行


new_sheet()

total_rows = 0
start = time.time()

############################################################
# 执行读写操作
############################################################
try:
    for chunk in pd.read_sql_query(sql, engine, chunksize=READ_CHUNK_SIZE):
        # NaN -> None，避免写进 Excel 后变成 nan 字符串或 #NUM!
        chunk = chunk.astype(object).where(pd.notna(chunk), None)

        for row in chunk.itertuples(index=False, name=None):
            if state["row_in_sheet"] >= SHEET_ROW_LIMIT:
                new_sheet()

            state["worksheet"].write_row(state["row_in_sheet"], 0, row)
            state["row_in_sheet"] += 1
            total_rows += 1

            if total_rows % 100_000 == 0:
                elapsed = time.time() - start
                rate = total_rows / elapsed if elapsed else 0
                print(f"[{elapsed:8.1f}s] 已写入 {total_rows:,} 行 "
                      f"({rate:,.0f} 行/秒)，"
                      f"当前 Sheet: Data_Part_{state['sheet_index']}")
        del chunk
finally:
    workbook.close()

print(f"完成：{total_rows:,} 行，文件 {OUTPUT_FILE}，"
      f"总耗时 {time.time() - start:.1f}s")