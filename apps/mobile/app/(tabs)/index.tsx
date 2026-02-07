import { Redirect } from "expo-router";

// default view is timeline
export default function Index() {
    return <Redirect href="/timeline" />;
}
