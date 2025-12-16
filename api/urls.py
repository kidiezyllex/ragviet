"""
API URL Configuration
"""
from django.urls import path
from . import views

urlpatterns = [
    # Authentication endpoints
    path('auth/login/', views.LoginView.as_view(), name='login'),
    path('auth/register/', views.RegisterView.as_view(), name='register'),
    path('auth/logout/', views.LogoutView.as_view(), name='logout'),
    path('auth/forgot-password/', views.ForgotPasswordView.as_view(), name='forgot-password'),
    path('auth/reset-password/', views.ResetPasswordView.as_view(), name='reset-password'),
    path('auth/verify-session/', views.VerifySessionView.as_view(), name='verify-session'),
    
    # Chat endpoints
    path('chat/send/', views.ChatSendView.as_view(), name='chat-send'),
    path('chat/sessions/', views.ChatSessionsView.as_view(), name='chat-sessions'),
    path('chat/sessions/create/', views.CreateChatSessionView.as_view(), name='create-chat-session'),
    path('chat/history/<str:session_id>/', views.ChatHistoryView.as_view(), name='chat-history'),
    
    # File management endpoints
    path('files/upload/', views.FileUploadView.as_view(), name='file-upload'),
    path('files/list/', views.FileListView.as_view(), name='file-list'),
    path('files/delete/', views.FileDeleteView.as_view(), name='file-delete'),
    path('files/clear-all/', views.FileClearAllView.as_view(), name='file-clear-all'),
    path('files/view/<str:filename>/', views.FileViewView.as_view(), name='file-view'),

    # Admin endpoints
    path('admin/users/', views.AdminUsersView.as_view(), name='admin-users'),
    path('admin/files/', views.AdminFilesView.as_view(), name='admin-files'),
    path('admin/users/status/', views.AdminUserStatusView.as_view(), name='admin-user-status'),
    path('admin/users/delete/', views.AdminUserDeleteView.as_view(), name='admin-user-delete'),
    path('admin/files/delete/', views.AdminFileDeleteView.as_view(), name='admin-file-delete'),
    path('admin/files/download-log/', views.AdminFileDownloadLogView.as_view(), name='admin-file-download-log'),
]

