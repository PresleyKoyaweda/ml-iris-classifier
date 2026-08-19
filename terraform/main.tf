terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "project_name" {
  default = "ml-demo"
}

variable "environment" {
  default = "dev"
}

variable "location" {
  default = "eastus"
}

data "azurerm_client_config" "current" {}

resource "azurerm_resource_group" "rg" {
  name     = "rg-${var.project_name}-${var.environment}"
  location = var.location

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "azurerm_storage_account" "storage" {
  name                     = "st${replace(var.project_name, "-", "")}${var.environment}"
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_application_insights" "ai" {
  name                = "ai-${var.project_name}"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  application_type    = "web"
}

resource "azurerm_key_vault" "kv" {
  name                = "kv-${var.project_name}-${var.environment}"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    key_permissions = ["Get", "List"]
    secret_permissions = ["Get", "List", "Set"]
  }
}

resource "azurerm_machine_learning_workspace" "mlw" {
  name                    = "mlw-${var.project_name}"
  location                = azurerm_resource_group.rg.location
  resource_group_name     = azurerm_resource_group.rg.name
  application_insights_id = azurerm_application_insights.ai.id
  key_vault_id            = azurerm_key_vault.kv.id
  storage_account_id      = azurerm_storage_account.storage.id

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_container_registry" "acr" {
  name                = "acr${replace(var.project_name, "-", "")}${var.environment}"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Standard"
  admin_enabled       = true
}

output "workspace_name" {
  value = azurerm_machine_learning_workspace.mlw.name
}

output "resource_group_name" {
  value = azurerm_resource_group.rg.name
}

output "container_registry_url" {
  value = azurerm_container_registry.acr.login_server
}
