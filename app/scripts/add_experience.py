from src.database import container

experience = {
    "id": "cmha-experience",
    "type": "experience",
    "organization": "Canadian Mental Health Association",
    "role": "IT Intern",
    "period": "May 2025 – August 2025",
    "highlights": [
        "Developed booking workflows with Power Apps and Power Automate",
        "Eliminated double bookings and saved over five staff hours weekly",
        "Provided IT support for over 100 staff members"
    ],
    "technologies": [
        "Power Apps",
        "Power Automate",
        "SharePoint"
    ],
    "isPublic": True
}

container.upsert_item(experience)

print("Experience record saved!")