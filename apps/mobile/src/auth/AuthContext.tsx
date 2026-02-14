import React, { createContext, useContext, useMemo, useState } from "react";

import {
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
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string, name?: string) => Promise<void>;
  logout: () => Promise<void>;
  hydrateFromToken: (token: string) => Promise<void>;
  updateName: (name: string) => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<AuthUser | null>(null);

  async function login(username: string, password: string) {
    const auth = await loginApi({ username, password });
    setAuthToken(auth.token);
    setToken(auth.token);
    setUser(auth.user);
  }

  async function register(username: string, password: string, name?: string) {
    const auth = await registerApi({ username, password, name });
    setAuthToken(auth.token);
    setToken(auth.token);
    setUser(auth.user);
  }

  async function logout() {
    try {
      await logoutApi();
    } finally {
      setAuthToken(null);
      setToken(null);
      setUser(null);
    }
  }

  async function hydrateFromToken(nextToken: string) {
    setAuthToken(nextToken);
    setToken(nextToken);
    const me = await fetchMe();
    setUser(me);
  }

  async function updateName(name: string) {
    const next = await updateMe({ name });
    setUser(next);
  }

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      token,
      isAuthenticated: Boolean(user && token),
      login,
      register,
      logout,
      hydrateFromToken,
      updateName,
    }),
    [user, token]
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
