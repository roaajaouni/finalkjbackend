# Generated by Django 5.0.4 on 2024-06-05 12:14

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
        ('mother', '0005_remove_report_important_notice_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='mother',
            name='group',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='auth.group'),
        ),
    ]
