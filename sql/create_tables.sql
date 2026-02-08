-- ============================================================================
-- FDA Database Schema
-- ============================================================================

USE FDADatabase;
GO

-- Create schema for analytics tables
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'model')
BEGIN
    EXEC('CREATE SCHEMA model');
END
GO

-- ============================================================================
-- TABLE 1: MAUDE (Adverse Events)
-- ============================================================================
IF OBJECT_ID('dbo.MAUDE', 'U') IS NULL
BEGIN

CREATE TABLE [dbo].[MAUDE](
	[MDR_REPORT_KEY] [nvarchar](max) NULL,
	[EVENT_KEY] [nvarchar](max) NULL,
	[REPORT_NUMBER] [nvarchar](max) NULL,
	[REPORT_SOURCE_CODE] [nvarchar](max) NULL,
	[MANUFACTURER_LINK_FLAG_] [nvarchar](max) NULL,
	[NUMBER_DEVICES_IN_EVENT] [nvarchar](max) NULL,
	[NUMBER_PATIENTS_IN_EVENT] [nvarchar](max) NULL,
	[DATE_RECEIVED] [date] NULL,
	[ADVERSE_EVENT_FLAG] [nvarchar](max) NULL,
	[PRODUCT_PROBLEM_FLAG] [nvarchar](max) NULL,
	[DATE_REPORT] [date] NULL,
	[DATE_OF_EVENT] [date] NULL,
	[REPROCESSED_AND_REUSED_FLAG] [nvarchar](max) NULL,
	[REPORTER_OCCUPATION_CODE] [nvarchar](max) NULL,
	[HEALTH_PROFESSIONAL] [nvarchar](max) NULL,
	[INITIAL_REPORT_TO_FDA] [nvarchar](max) NULL,
	[DATE_FACILITY_AWARE] [date] NULL,
	[REPORT_DATE] [date] NULL,
	[REPORT_TO_FDA] [nvarchar](max) NULL,
	[DATE_REPORT_TO_FDA] [date] NULL,
	[EVENT_LOCATION] [nvarchar](max) NULL,
	[DATE_REPORT_TO_MANUFACTURER] [date] NULL,
	[MANUFACTURER_CONTACT_T_NAME] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_F_NAME] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_L_NAME] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_STREET_1] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_STREET_2] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_CITY] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_STATE] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_ZIP_CODE] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_ZIP_EXT] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_COUNTRY] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_POSTAL] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_AREA_CODE] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_EXCHANGE] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_PHONE_NO] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_EXTENSION] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_PCOUNTRY] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_PCITY] [nvarchar](max) NULL,
	[MANUFACTURER_CONTACT_PLOCAL] [nvarchar](max) NULL,
	[MANUFACTURER_G1_NAME] [nvarchar](max) NULL,
	[MANUFACTURER_G1_STREET_1] [nvarchar](max) NULL,
	[MANUFACTURER_G1_STREET_2] [nvarchar](max) NULL,
	[MANUFACTURER_G1_CITY] [nvarchar](max) NULL,
	[MANUFACTURER_G1_STATE_CODE] [nvarchar](max) NULL,
	[MANUFACTURER_G1_ZIP_CODE] [nvarchar](max) NULL,
	[MANUFACTURER_G1_ZIP_CODE_EXT] [nvarchar](max) NULL,
	[MANUFACTURER_G1_COUNTRY_CODE] [nvarchar](max) NULL,
	[MANUFACTURER_G1_POSTAL_CODE] [nvarchar](max) NULL,
	[DATE_MANUFACTURER_RECEIVED] [date] NULL,
	[DEVICE_DATE_OF_MANUFACTURE] [date] NULL,
	[SINGLE_USE_FLAG] [nvarchar](max) NULL,
	[REMEDIAL_ACTION] [nvarchar](max) NULL,
	[PREVIOUS_USE_CODE] [nvarchar](max) NULL,
	[REMOVAL_CORRECTION_NUMBER] [nvarchar](max) NULL,
	[EVENT_TYPE] [nvarchar](max) NULL,
	[DISTRIBUTOR_NAME] [nvarchar](max) NULL,
	[DISTRIBUTOR_ADDRESS_1] [nvarchar](max) NULL,
	[DISTRIBUTOR_ADDRESS_2] [nvarchar](max) NULL,
	[DISTRIBUTOR_CITY] [nvarchar](max) NULL,
	[DISTRIBUTOR_STATE_CODE] [nvarchar](max) NULL,
	[DISTRIBUTOR_ZIP_CODE] [nvarchar](max) NULL,
	[DISTRIBUTOR_ZIP_CODE_EXT] [nvarchar](max) NULL,
	[REPORT_TO_MANUFACTURER] [nvarchar](max) NULL,
	[MANUFACTURER_NAME] [nvarchar](max) NULL,
	[MANUFACTURER_ADDRESS_1] [nvarchar](max) NULL,
	[MANUFACTURER_ADDRESS_2] [nvarchar](max) NULL,
	[MANUFACTURER_CITY] [nvarchar](max) NULL,
	[MANUFACTURER_STATE_CODE] [nvarchar](max) NULL,
	[MANUFACTURER_ZIP_CODE] [nvarchar](max) NULL,
	[MANUFACTURER_ZIP_CODE_EXT] [nvarchar](max) NULL,
	[MANUFACTURER_COUNTRY_CODE] [nvarchar](max) NULL,
	[MANUFACTURER_POSTAL_CODE] [nvarchar](max) NULL,
	[TYPE_OF_REPORT] [nvarchar](max) NULL,
	[SOURCE_TYPE] [nvarchar](max) NULL,
	[DATE_ADDED] [date] NULL,
	[DATE_CHANGED] [date] NULL,
	[REPORTER_COUNTRY_CODE] [nvarchar](max) NULL,
	[PMA_PMN_NUM] [nvarchar](max) NULL,
	[EXEMPTION_NUMBER] [nvarchar](max) NULL,
	[SUMMARY_REPORT] [nvarchar](max) NULL,
	[NOE_SUMMARIZED] [nvarchar](max) NULL,
	[SUPPL_DATES_FDA_RECEIVED] [date] NULL,
	[SUPPL_DATES_MFR_RECEIVED] [date] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
END
GO

-- ============================================================================
-- TABLE 2: Premarket510k
-- ============================================================================
IF OBJECT_ID('dbo.Premarket510k', 'U') IS NULL
BEGIN
CREATE TABLE [dbo].[Premarket510k](
	[KNUMBER] [nvarchar](20) NULL,
	[APPLICANT] [varchar](500) NULL,
	[CONTACT] [varchar](500) NULL,
	[STREET1] [varchar](500) NULL,
	[STREET2] [varchar](500) NULL,
	[CITY] [varchar](255) NULL,
	[STATE] [nvarchar](50) NULL,
	[COUNTRY_CODE] [nvarchar](50) NULL,
	[ZIP] [nvarchar](20) NULL,
	[POSTAL_CODE] [nvarchar](20) NULL,
	[DATERECEIVED] [date] NULL,
	[DECISIONDATE] [date] NULL,
	[DECISION] [varchar](255) NULL,
	[REVIEWADVISECOMM] [nvarchar](100) NULL,
	[PRODUCTCODE] [nvarchar](50) NULL,
	[STATEORSUMM] [varchar](max) NULL,
	[CLASSADVISECOMM] [nvarchar](100) NULL,
	[SSPINDICATOR] [nvarchar](20) NULL,
	[TYPE] [nvarchar](50) NULL,
	[THIRDPARTY] [nvarchar](50) NULL,
	[EXPEDITEDREVIEW] [nvarchar](50) NULL,
	[DEVICENAME] [varchar](1000) NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
END
GO

-- ============================================================================
-- TABLE 3: PMA (Premarket Approval)
IF OBJECT_ID('dbo.PMA', 'U') IS NULL
BEGIN
CREATE TABLE [dbo].[PMA](
	[PMANUMBER] [nvarchar](20) NULL,
	[SUPPLEMENTNUMBER] [nvarchar](20) NULL,
	[APPLICANT] [nvarchar](500) NULL,
	[STREET_1] [varchar](500) NULL,
	[STREET_2] [varchar](500) NULL,
	[CITY] [varchar](255) NULL,
	[STATE] [nvarchar](50) NULL,
	[ZIP] [nvarchar](20) NULL,
	[ZIP_EXT] [nvarchar](10) NULL,
	[GENERICNAME] [nvarchar](500) NULL,
	[TRADENAME] [nvarchar](500) NULL,
	[PRODUCTCODE] [nvarchar](50) NULL,
	[ADVISORYCOMMITTEE] [nvarchar](255) NULL,
	[SUPPLEMENTTYPE] [nvarchar](100) NULL,
	[SUPPLEMENTREASON] [varchar](max) NULL,
	[REVIEWGRANTEDYN] [nvarchar](5) NULL,
	[DATERECEIVED] [date] NULL,
	[DECISIONDATE] [date] NULL,
	[DOCKETNUMBER] [nvarchar](50) NULL,
	[FEDREGNOTICEDATE] [date] NULL,
	[DECISIONCODE] [nvarchar](50) NULL,
	[AOSTATEMENT] [nvarchar](max) NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
END
GO
-- ============================================================================
-- TABLE 4: Productcode
IF OBJECT_ID('dbo.ProuductCode', 'U') IS NULL
BEGIN
CREATE TABLE [dbo].[ProductCode](
	[REVIEW_PANEL] [nvarchar](100) NULL,
	[MEDICALSPECIALTY] [nvarchar](100) NULL,
	[PRODUCTCODE] [nvarchar](50) NULL,
	[DEVICENAME] [nvarchar](max) NULL,
	[DEVICECLASS] [nvarchar](20) NULL,
	[UNCLASSIFIED_REASON] [varchar](255) NULL,
	[GMPEXEMPTFLAG] [nvarchar](10) NULL,
	[THIRDPARTYFLAG] [nvarchar](10) NULL,
	[REVIEWCODE] [nvarchar](20) NULL,
	[REGULATIONNUMBER] [nvarchar](50) NULL,
	[SUBMISSION_TYPE_ID] [nvarchar](20) NULL,
	[DEFINITION] [nvarchar](max) NULL,
	[PHYSICALSTATE] [nvarchar](max) NULL,
	[TECHNICALMETHOD] [nvarchar](max) NULL,
	[TARGETAREA] [nvarchar](max) NULL,
	[Implant_Flag] [nvarchar](10) NULL,
	[Life_Sustain_support_flag] [nvarchar](10) NULL,
	[SummaryMalfunctionReporting] [nvarchar](10) NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
END
GO

-- ============================================================================
-- TABLE 5: recall
IF OBJECT_ID('dbo.Recall', 'U') IS NULL
BEGIN
CREATE TABLE [dbo].[RECALL](
	[cfres_id] [nvarchar](255) NULL,
	[product_res_number] [nvarchar](255) NULL,
	[event_date_initiated] [date] NULL,
	[event_date_posted] [date] NULL,
	[recall_status] [nvarchar](255) NULL,
	[res_event_number] [nvarchar](255) NULL,
	[product_code] [nvarchar](255) NULL,
	[k_numbers] [nvarchar](255) NULL,
	[product_description] [nvarchar](max) NULL,
	[code_info] [nvarchar](max) NULL,
	[recalling_firm] [nvarchar](max) NULL,
	[address_1] [nvarchar](max) NULL,
	[address_2] [nvarchar](max) NULL,
	[city] [nvarchar](max) NULL,
	[state] [nvarchar](max) NULL,
	[postal_code] [nvarchar](max) NULL,
	[additional_info_contact] [nvarchar](max) NULL,
	[reason_for_recall] [nvarchar](max) NULL,
	[root_cause_description] [nvarchar](max) NULL,
	[action] [nvarchar](max) NULL,
	[product_quantity] [nvarchar](max) NULL,
	[distribution_pattern] [nvarchar](max) NULL,
	[openfda] [nvarchar](max) NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
END
GO

PRINT 'Tables created successfully!';
