-- ============================================================
-- 智慧养蜂项目 —— 蜂产品溯源系统数据库
-- （对应成员3/溯源负责人；与主系统共用MySQL实例即可）
--
-- 设计要点：
--   1. 溯源码全局唯一，消费者凭码查询全链路记录
--   2. 每条记录含 prev_hash + data_hash 形成"哈希链"，
--      任一环节被事后篡改，链校验立即失败（轻量级防篡改，
--      答辩可表述为"类区块链的可信溯源链"）
-- ============================================================

CREATE DATABASE IF NOT EXISTS bee_trace
  DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE bee_trace;

-- 1. 蜂场/产地登记表
CREATE TABLE IF NOT EXISTS bee_farm (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    farm_name   VARCHAR(100)  NOT NULL COMMENT '蜂场名称',
    beekeeper   VARCHAR(50)   NOT NULL COMMENT '蜂农姓名',
    province    VARCHAR(30)   NOT NULL COMMENT '省',
    city        VARCHAR(30)   NOT NULL COMMENT '市',
    county      VARCHAR(30)   DEFAULT '' COMMENT '县/区',
    latitude    DECIMAL(10,6) NOT NULL COMMENT '纬度',
    longitude   DECIMAL(10,6) NOT NULL COMMENT '经度',
    apiaries    INT           DEFAULT 50 COMMENT '蜂箱数量',
    created_at  DATETIME      DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB COMMENT='蜂场产地登记';

-- 2. 蜂蜜批次表（一次采蜜/一个产品批次 = 一个溯源码）
CREATE TABLE IF NOT EXISTS honey_batch (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    trace_code    VARCHAR(24)  NOT NULL COMMENT '溯源码，消费者凭此查询',
    farm_id       INT          NOT NULL COMMENT '关联蜂场',
    batch_no      VARCHAR(30)  NOT NULL COMMENT '批次号',
    honey_type    VARCHAR(30)  NOT NULL COMMENT '蜜种：洋槐/枣花/荆条...',
    produce_date  DATE         NOT NULL COMMENT '采蜜日期',
    weight_kg     DECIMAL(8,2) NOT NULL COMMENT '批次重量(kg)',
    merchant      VARCHAR(100) NOT NULL COMMENT '加工/销售商家',
    quality_grade VARCHAR(20)  DEFAULT '' COMMENT '质量等级(一级/合格)',
    quality_report VARCHAR(255) DEFAULT '' COMMENT '质检报告编号',
    prev_hash     CHAR(64)     NOT NULL COMMENT '链上前一记录哈希',
    data_hash     CHAR(64)     NOT NULL COMMENT '本记录内容哈希',
    created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_trace_code (trace_code),
    KEY idx_farm (farm_id),
    CONSTRAINT fk_batch_farm FOREIGN KEY (farm_id) REFERENCES bee_farm(id)
) ENGINE=InnoDB COMMENT='蜂蜜产品批次';

-- 3. 溯源环节事件表（产地→采蜜→质检→加工→物流→销售）
CREATE TABLE IF NOT EXISTS trace_event (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    batch_id   INT          NOT NULL,
    stage      ENUM('FARMING','HARVEST','QUALITY','PROCESSING','LOGISTICS','SALE')
               NOT NULL COMMENT '环节类型',
    title      VARCHAR(100) NOT NULL COMMENT '环节标题',
    detail     TEXT         COMMENT '详细信息(JSON)：时间/地点/操作人等',
    operator   VARCHAR(50)  NOT NULL COMMENT '录入人',
    prev_hash  CHAR(64)     NOT NULL COMMENT '链上前一记录哈希',
    data_hash  CHAR(64)     NOT NULL COMMENT '本记录内容哈希',
    created_at DATETIME     DEFAULT CURRENT_TIMESTAMP,
    KEY idx_batch (batch_id),
    CONSTRAINT fk_event_batch FOREIGN KEY (batch_id) REFERENCES honey_batch(id)
) ENGINE=InnoDB COMMENT='溯源环节事件链';

-- ============================================================
-- 示例数据（可选执行）
-- ============================================================
-- INSERT INTO bee_farm (farm_name, beekeeper, province, city, county,
--     latitude, longitude, apiaries)
-- VALUES ('秦岭洋槐蜂场', '张三', '陕西', '宝鸡', '太白县',
--     34.058300, 107.319700, 120);
