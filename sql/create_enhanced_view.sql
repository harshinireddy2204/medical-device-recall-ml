-- ============================================================================
-- Enhanced View for Dashboard - Combines RPSS with Device Details
-- ============================================================================

USE FDADatabase;
GO

IF OBJECT_ID('dbo.vw_device_rpss_categorized', 'V') IS NOT NULL
    DROP VIEW dbo.vw_device_rpss_categorized;
GO

CREATE VIEW dbo.vw_device_rpss_categorized AS
SELECT 
    PMA_PMN_NUM,
    rpss,
    rpss_category,
    recall_count,
    total_adverse_events,
    unique_manufacturers,
    device_class,
    root_cause_description,
    last_scored
FROM model.device_rpss;
GO

PRINT 'Enhanced view created successfully!';