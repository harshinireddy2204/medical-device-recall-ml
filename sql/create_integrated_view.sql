-- ============================================================================
-- FDA Integrated View - Combines all data sources
-- ============================================================================

USE FDADatabase;
GO

-- Drop existing view if it exists
IF OBJECT_ID('dbo.vw_FDA_Device_Integrated', 'V') IS NOT NULL
    DROP VIEW dbo.vw_FDA_Device_Integrated;
GO

-- Create integrated view
CREATE VIEW dbo.vw_FDA_Device_Integrated AS
SELECT 
    -- Device identifiers
    COALESCE(pma.pma_number, k.k_number, r.product_number) AS PMA_PMN_NUM,
    
    -- PMA data
    pma.pma_number,
    pma.product_code AS pma_product_code,
    pma.applicant AS pma_applicant,
    pma.generic_name AS GENERICNAME,
    pma.trade_name AS TRADENAME,
    
    -- 510k data
    k.k_number,
    k.product_code AS k_product_code,
    k.applicant AS k_applicant,
    k.device_name AS k_devicename,
    k.k_type,
    
    -- Adverse events (MAUDE) aggregated
    COUNT(DISTINCT m.mdr_report_key) AS adverse_event_count,
    SUM(m.adverse_event_flag) AS total_adverse_flag,
    SUM(m.product_problem_flag) AS total_product_problem_flag,
    MIN(m.event_date) AS first_event_date,
    MAX(m.event_date) AS last_event_date,
    
    -- Product classification
    pc.device_name AS pc_devicename,
    pc.device_class AS DEVICECLASS,
    pc.medical_specialty AS MEDICALSPECIALTY,
    
    -- Recall data
    r.res_event_number AS cfres_id,
    r.recall_status,
    r.event_date_initiated,
    r.reason_for_recall,
    r.root_cause_description
    
FROM dbo.PMA pma
FULL OUTER JOIN dbo.Premarket510k k 
    ON pma.product_code = k.product_code
LEFT JOIN dbo.MAUDE m 
    ON (m.device_identifier = pma.pma_number OR m.device_identifier = k.k_number)
LEFT JOIN dbo.Productcode pc 
    ON (pc.product_code = pma.product_code OR pc.product_code = k.product_code)
LEFT JOIN dbo.recall r 
    ON (r.product_number = pma.pma_number OR r.product_number = k.k_number)
    
GROUP BY
    COALESCE(pma.pma_number, k.k_number, r.product_number),
    pma.pma_number, pma.product_code, pma.applicant, 
    pma.generic_name, pma.trade_name,
    k.k_number, k.product_code, k.applicant, k.device_name, k.k_type,
    pc.device_name, pc.device_class, pc.medical_specialty,
    r.res_event_number, r.recall_status, r.event_date_initiated,
    r.reason_for_recall, r.root_cause_description;
GO

PRINT 'Integrated view created successfully!';