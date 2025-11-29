"""
Custom authentication for Django REST Framework
"""
from rest_framework import authentication
from rest_framework.exceptions import AuthenticationFailed
from utils.auth import AuthManager
from utils.database import Database


class SessionTokenAuthentication(authentication.BaseAuthentication):
    """
    Custom authentication using session tokens from AuthManager
    """
    
    def authenticate(self, request):
        # Get token from Authorization header or session
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')
        
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        else:
            # Try to get from request data or query params
            token = request.data.get('session_id') or request.query_params.get('session_id')
        
        if not token:
            return None
        
        # Initialize auth manager
        try:
            database = Database()
            auth_manager = AuthManager(database)
            user = auth_manager.get_user_from_session(token)
            
            if user:
                return (user, token)
            else:
                raise AuthenticationFailed('Invalid or expired session token')
        except Exception as e:
            raise AuthenticationFailed(f'Authentication failed: {str(e)}')

