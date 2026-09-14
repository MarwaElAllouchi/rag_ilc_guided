from app.core.config import get_settings

settings = get_settings()

print(settings.app_name)
print(settings.database_url)
print(settings.raw_data_path)