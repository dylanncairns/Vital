import React, { createContext, useContext, useMemo, useState } from "react";

import {
  deleteMe,
  AuthUser,
  fetchMe,
  login as loginApi,
  logout as logoutApi,
  register as registerApi,
  setAuthToken,
  updateMe,
} from "../api/client";

type AuthContextValue = {
  user: AuthUser | null;
  token: string | null;
  isAuthenticated: boolean;
  isHydrated: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string, name?: string) => Promise<void>;
  logout: () => Promise<void>;
  hydrateFromToken: (token: string) => Promise<void>;
  updateName: (name: string) => Promise<void>;
  deleteAccount: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);
const AUTH_TOKEN_STORAGE_KEY = "vital.auth.token";

function readStoredToken(): string | null {
  try {
    const storage = globalThis?.localStorage;
    if (!storage) return null;
    const value = storage.getItem(AUTH_TOKEN_STORAGE_KEY);
    return value && value.trim() ? value : null;
  } catch {
    return null;
  }
}

function writeStoredToken(token: string | null): void {
  try {
    const storage = globalThis?.localStorage;
    if (!storage) return;
    if (token && token.trim()) {
      storage.setItem(AUTH_TOKEN_STORAGE_KEY, token);
    } else {
      storage.removeItem(AUTH_TOKEN_STORAGE_KEY);
    }
  } catch {
    // no-op
  }
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isHydrated, setIsHydrated] = useState(false);

  React.useEffect(() => {
    let cancelled = false;
    async function hydrate() {
      const storedToken = readStoredToken();
      if (!storedToken) {
        if (!cancelled) setIsHydrated(true);
        return;
      }
      try {
        setAuthToken(storedToken);
        const me = await fetchMe();
        if (cancelled) return;
        setToken(storedToken);
        setUser(me);
      } catch {
        if (cancelled) return;
        setAuthToken(null);
        setToken(null);
        setUser(null);
        writeStoredToken(null);
      } finally {
        if (!cancelled) setIsHydrated(true);
      }
    }
    void hydrate();
    return () => {
      cancelled = true;
    };
  }, []);

  async function login(username: string, password: string) {
    const auth = await loginApi({ username, password });
    setAuthToken(auth.token);
    setToken(auth.token);
    setUser(auth.user);
    writeStoredToken(auth.token);
  }

  async function register(username: string, password: string, name?: string) {
    const auth = await registerApi({ username, password, name });
    setAuthToken(auth.token);
    setToken(auth.token);
    setUser(auth.user);
    writeStoredToken(auth.token);
  }

  async function logout() {
    try {
      await logoutApi();
    } finally {
      setAuthToken(null);
      setToken(null);
      setUser(null);
      writeStoredToken(null);
    }
  }

  async function hydrateFromToken(nextToken: string) {
    setAuthToken(nextToken);
    setToken(nextToken);
    const me = await fetchMe();
    setUser(me);
    writeStoredToken(nextToken);
  }

  async function updateName(name: string) {
    const next = await updateMe({ name });
    setUser(next);
  }

  async function deleteAccount() {
    try {
      await deleteMe();
    } finally {
      setAuthToken(null);
      setToken(null);
      setUser(null);
      writeStoredToken(null);
    }
  }

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      token,
      isAuthenticated: Boolean(user && token),
      isHydrated,
      login,
      register,
      logout,
      hydrateFromToken,
      updateName,
      deleteAccount,
    }),
    [isHydrated, user, token]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return ctx;
}
