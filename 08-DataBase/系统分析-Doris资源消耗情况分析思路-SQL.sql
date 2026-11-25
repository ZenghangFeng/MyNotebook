-- 单个物化视图查询时间分析
SELECT *
  FROM tasks("type"="mv")
 WHERE MvName = 'mv_production_forging_record'
 ORDER BY CreateTime DESC
 LIMIT 10;


-- 工作负载分析与诊断
-- https://doris.apache.org/zh-CN/docs/3.x/admin-manual/workload-management/analysis-diagnosis


-- information_schema.active_queries（查询当前正在执行的查询）：
SELECT query_id, sql 
FROM information_schema.active_queries 
WHERE query_id = 'your_query_id';


-- __internal_schema.audit_log（查询历史已完成的查询，需先开启 enable_audit_plugin）：
-- SET GLOBAL enable_audit_plugin = true;

SELECT query_id, stmt 
FROM __internal_schema.audit_log 
WHERE query_id = 'your_query_id';


-- information_schema.processlist（查看当前连接及其正在执行的 SQL）：
SELECT id, user, info 
FROM information_schema.processlist 
WHERE query_id = 'your_query_id';