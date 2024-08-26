# Generated by Django 4.2.7 on 2024-04-10 17:57

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Mother',
            fields=[
                ('username', models.CharField(blank=True, max_length=200, null=True)),
                ('name', models.CharField(blank=True, max_length=200, null=True)),
                ('email', models.CharField(blank=True, max_length=200, null=True)),
                ('phone', models.CharField(blank=True, max_length=200, null=True)),
                ('address', models.CharField(blank=True, max_length=200, null=True)),
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False, unique=True)),
                ('user', models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Child',
            fields=[
                ('name', models.CharField(blank=True, max_length=200, null=True)),
                ('age', models.CharField(blank=True, choices=[('5', '5'), ('6', '6')], max_length=200, null=True)),
                ('child_gender', models.CharField(blank=True, choices=[('male', 'Male'), ('female', 'Female')], max_length=200, null=True)),
                ('featured_image', models.ImageField(blank=True, default='default.png', null=True, upload_to='staticfiles/images/')),
                ('meal', models.CharField(blank=True, choices=[('meat', 'Meat'), ('milk', 'milk')], max_length=200, null=True)),
                ('notes', models.CharField(blank=True, max_length=200, null=True)),
                ('state_health', models.CharField(blank=True, max_length=200, null=True)),
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False, unique=True)),
                ('mom', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='mother.mother')),
            ],
        ),
    ]
