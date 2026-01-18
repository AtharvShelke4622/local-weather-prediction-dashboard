from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta

from database import AsyncSessionLocal
import crud
from schemas import LoginRequest, RegisterRequest, Token, UserOut
from auth import (
    verify_password, 
    create_access_token, 
    get_password_hash,
    get_current_active_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from db_models import User

router = APIRouter(prefix="/auth", tags=["authentication"])

async def get_session() -> AsyncSession:
    """Get database session."""
    async with AsyncSessionLocal() as session:
        yield session

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    user_data: RegisterRequest,
    session: AsyncSession = Depends(get_session)
):
    """Register a new user."""
    # Validate passwords match
    if user_data.password != user_data.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match"
        )
    
    # Check if user already exists
    existing_user = await crud.get_user_by_username(session, user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    existing_email = await crud.get_user_by_email(session, user_data.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user = await crud.create_user(
        session=session,
        email=user_data.email,
        username=user_data.username,
        password=user_data.password,
        role=user_data.role or "user"
    )
    
    await session.commit()
    
    return {"message": "User registered successfully", "user_id": user.id}

@router.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    session: AsyncSession = Depends(get_session)
):
    """Authenticate user and return access token."""
    # Authenticate user
    user = await crud.get_user_by_username(session, login_data.username)
    
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Update last login
    await crud.update_user_last_login(session, user.id)
    await session.commit()
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.id}, 
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # Convert to seconds
    }

@router.get("/me", response_model=UserOut)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information."""
    return current_user

@router.get("/users", response_model=list[UserOut])
async def list_users(
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session)
):
    """List all users (admin only)."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    users = await crud.list_users(session)
    return [UserOut.model_validate(user) for user in users]